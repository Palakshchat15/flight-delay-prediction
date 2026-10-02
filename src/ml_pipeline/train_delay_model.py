"""
Flight arrival-delay classifier, predicted 2 HOURS BEFORE SCHEDULED DEPARTURE:
will this flight arrive 15+ minutes late (ArrDel15)?

Prediction point: cutoff = scheduled departure - 2 h. Features are everything known at that
moment and nothing later (README_PIPELINE.md, "Features and their timing rules"):
  * schedule + previous-calendar-month history (the old "weeks ahead" feature set)
  * the inbound aircraft (same tail number): previous leg's departure / arrival delay only
    if it had actually departed / arrived before the cutoff, "scheduled to leave but has not
    left yet" (overdue minutes), projected turn slack, aircraft's latest known delay
  * congestion in the 2 h before the cutoff: origin / destination / carrier / network delays,
    flights still waiting to leave the origin
  * weather: origin at the cutoff hour; destination at the scheduled arrival hour (observed
    weather standing in for a forecast)
Built by cutoff_features.py -> warehouse.flight_cutoff_features and checked by
leakage_check.py (event times < cutoff) before this script runs.

Time-ordered split over the last MODEL_WINDOW_MONTHS (4) months of data, relative to the
latest month (so it moves forward as new BTS months arrive):
  month 1  history only: feeds the previous-month rate features of month 2
  month 2  stage 1 fit
  month 3  VALIDATION: design choices (design_checks.py), inbound-rule threshold, model comparison
  month 4  TEST: scored once by models refit on months 2-3

Models (all compared on the same test month):
  1. Historical-rate rule: carrier+route delay-rate lift of the previous month.
  2. Inbound-delay rule: the inbound aircraft's known delay at the cutoff (minutes). As a
     yes/no rule it flags flights whose inbound is already >= X min late, X chosen on the
     validation month (best F1).
  3. Logistic regression (unweighted; median imputation + missing indicators, standardised).
  4. LightGBM (unweighted binary log-loss).
Operating points: every model flags the riskiest 20% and the riskiest 10% of the scored
month's flights (equal flag rates -> directly comparable precision/recall); threshold-free
AUC-ROC / AUC-PR as well.

Reproducibility: ORDER BY flight_date, flight_key; seeded models; LightGBM deterministic=True,
force_row_wise=True, fixed thread count.

Outputs (outputs/): model_metrics.json, model_report.txt, shap_summary.png,
predicted_delays_test_month.csv (every flagged test flight + top SHAP reasons),
example_predictions.csv, model/lgbm_delay_model.txt; warehouse.model_test_predictions.
"""
import io
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_utils import (concat_columns, copy_text_column, get_engine, read_sql_chunks,  # noqa: E402
                      release_memory, shared_objects)
import cutoff_features as cf  # noqa: E402

OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
MODEL_DIR = OUTPUTS_DIR / "model"
SEED = 42
SMOOTHING = 50
FLAG_SHARE = 0.20            # main operating point: flag the riskiest 20%
FLAG_SHARES = (0.20, 0.10)   # also reported at 10%
MODEL_WINDOW_MONTHS = cf.MODEL_WINDOW_MONTHS
CLASS_WEIGHTED = False       # weeks-ahead design check: no validation gain, inflates scores
RELATIVE_RATES = True        # previous-month rates as lift (weeks-ahead design check)
SWAP_SIGNATURE_TURN_MIN = 20  # scheduled turn below this (or negative) = likely same-day aircraft swap
INBOUND_RULE_CANDIDATES = list(range(0, 181, 5))
LOAD_GROUP = 12              # columns per streamed query in load_data (memory only)

SCHEDULE_FEATURES = [
    "dep_hour", "arr_hour", "dep_minute_of_day", "day_of_week", "crs_elapsed_minutes",
    "distance_miles", "origin_sched_deps_in_hour", "dest_sched_arrs_in_hour",
    "origin_sched_deps_in_day",
]
ROTATION_KEY = ["carrier_code", "flight_number", "origin"]
ROTATION_FEATURES = {"hist_median_leg_of_day": "aircraft_leg_of_day",
                     "hist_median_turnaround": "sched_turnaround_minutes"}
RATE_GROUPS = {
    "hist_rate_carrier": ["carrier_code"], "hist_rate_origin": ["origin"], "hist_rate_dest": ["dest"],
    "hist_rate_route": ["route"], "hist_rate_carrier_route": ["carrier_code", "route"],
    "hist_rate_origin_hour": ["origin", "dep_hour"], "hist_rate_carrier_hour": ["carrier_code", "dep_hour"],
}
CATEGORICAL = ["carrier_code"]
WEEKS_AHEAD_NUMERIC = SCHEDULE_FEATURES + list(ROTATION_FEATURES) + list(RATE_GROUPS)
WEEKS_AHEAD_FEATURES = WEEKS_AHEAD_NUMERIC + CATEGORICAL
# Feature groups (design_checks.py judges each on the validation month)
FEATURE_GROUPS = {
    "inbound": cf.INBOUND_FEATURES,
    "congestion": cf.CONGESTION_FEATURES,
    "weather_origin": cf.WEATHER_ORIGIN_FEATURES,
    "weather_dest_forecast_standin": cf.WEATHER_DEST_FEATURES,
}
USED_GROUPS = ["inbound", "congestion", "weather_origin", "weather_dest_forecast_standin"]
NUMERIC = WEEKS_AHEAD_NUMERIC + [f for g in USED_GROUPS for f in FEATURE_GROUPS[g]]
FEATURES = NUMERIC + CATEGORICAL
# Same-day tail-number SCHEDULE features from the mart (no timing rule; they encode the aircraft
# that actually flew, i.e. delay-driven swaps): still excluded.
EXCLUDED_TAIL_FEATURES = ["aircraft_leg_of_day", "aircraft_legs_scheduled_today", "sched_turnaround_minutes"]

RULE = "Historical rate rule (carrier+route)"
INB_RULE = "Inbound-delay rule"
LOGREG = "Logistic regression"
LGBM = "LightGBM"
MODEL_NAMES = [RULE, INB_RULE, LOGREG, LGBM]
REF_LGBM = "LightGBM, weeks-ahead features only (reference)"

READABLE = {
    "dep_hour": "scheduled departure hour", "arr_hour": "scheduled arrival hour",
    "dep_minute_of_day": "scheduled departure time", "day_of_week": "day of week",
    "crs_elapsed_minutes": "scheduled block time", "distance_miles": "distance",
    "hist_median_leg_of_day": "flight's typical aircraft leg of day (prev. month)",
    "hist_median_turnaround": "flight's typical scheduled turnaround (prev. month, min)",
    "origin_sched_deps_in_hour": "origin departures scheduled that hour",
    "dest_sched_arrs_in_hour": "destination arrivals scheduled that hour",
    "origin_sched_deps_in_day": "origin departures scheduled that day",
    "hist_rate_carrier": "carrier's delay-rate lift (prev. month)",
    "hist_rate_origin": "origin's delay-rate lift (prev. month)",
    "hist_rate_dest": "destination's delay-rate lift (prev. month)",
    "hist_rate_route": "route's delay-rate lift (prev. month)",
    "hist_rate_carrier_route": "carrier+route delay-rate lift (prev. month)",
    "hist_rate_origin_hour": "origin+hour delay-rate lift (prev. month)",
    "hist_rate_carrier_hour": "carrier+hour delay-rate lift (prev. month)",
    "carrier_code": "carrier",
    "inb_has_prev": "aircraft has an earlier leg today",
    "inb_sched_turn_min": "scheduled turn after inbound leg (min)",
    "inb_prev_sched_dep_before_cutoff": "inbound leg scheduled to have left by cutoff",
    "inb_prev_departed": "inbound leg departed by cutoff",
    "inb_prev_dep_delay": "inbound leg departure delay (if departed by cutoff)",
    "inb_prev_arrived": "inbound leg arrived by cutoff",
    "inb_prev_arr_delay": "inbound leg arrival delay (if arrived by cutoff)",
    "inb_prev_overdue_min": "inbound leg overdue, not yet departed (min)",
    "inb_known_delay_min": "inbound aircraft known delay at cutoff (min)",
    "inb_projected_slack_min": "projected turn slack (min)",
    "ac_latest_dep_delay": "aircraft's latest known departure delay",
    "ac_latest_arr_delay": "aircraft's latest known arrival delay",
    "origin_recent_dep_count": "origin departures in 2 h before cutoff",
    "origin_recent_dep_delay_avg": "origin avg departure delay, 2 h before cutoff",
    "origin_pending_count": "origin flights overdue, not yet departed",
    "origin_pending_share": "share of origin flights overdue",
    "origin_recent_arr_delay_avg": "origin avg arrival delay, 2 h before cutoff",
    "dest_recent_arr_delay_avg": "destination avg arrival delay, 2 h before cutoff",
    "dest_recent_dep_delay_avg": "destination avg departure delay, 2 h before cutoff",
    "carrier_recent_dep_delay_avg": "carrier network avg departure delay, 2 h before cutoff",
    "nat_recent_dep_delay_avg": "US network avg departure delay, 2 h before cutoff",
    "wx_o_temp": "origin temperature at cutoff (C)", "wx_o_precip": "origin precipitation at cutoff (mm/h)",
    "wx_o_snow": "origin snowfall at cutoff (cm/h)", "wx_o_wind": "origin wind at cutoff (km/h)",
    "wx_o_gust": "origin gusts at cutoff (km/h)", "wx_o_cloud_low": "origin low cloud at cutoff (%)",
    "wx_o_precip_3h": "origin precipitation, 3 h to cutoff (mm)",
    "wx_d_temp": "destination temperature at sched. arrival (C)",
    "wx_d_precip": "destination precipitation at sched. arrival (mm/h)",
    "wx_d_snow": "destination snowfall at sched. arrival (cm/h)",
    "wx_d_wind": "destination wind at sched. arrival (km/h)",
    "wx_d_gust": "destination gusts at sched. arrival (km/h)",
    "wx_d_cloud_low": "destination low cloud at sched. arrival (%)",
}
LGB_PARAMS = dict(
    objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=SEED, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1,
)

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- data + features
def load_data(exclude_last_month=False):
    """Mart rows of the model window joined to the cutoff features, ORDER BY a stable key."""
    engine = get_engine()
    months = pd.read_sql("SELECT DISTINCT flight_year, flight_month FROM warehouse.mart_flight_features "
                         "ORDER BY 1, 2", engine).tail(MODEL_WINDOW_MONTHS)
    first = f"{int(months.iloc[0, 0])}-{int(months.iloc[0, 1]):02d}-01"
    if exclude_last_month:  # design_checks: the test month is never read
        last = months.iloc[-1]
        extra = f" AND flight_date < DATE '{int(last.iloc[0])}-{int(last.iloc[1]):02d}-01'"
    else:
        extra = ""
    rows = f"""
        FROM warehouse.mart_flight_features m
        JOIN warehouse.flight_cutoff_features c USING (flight_key)
        WHERE m.flight_date >= DATE '{first}'{extra}
        ORDER BY m.flight_date, m.flight_key"""
    # Server-side cursor + 200k-row chunks downcast to float32 as they arrive: a one-shot fetch of
    # ~2.2M x 75 columns as Python objects needs several GB and was OOM-killed in the shared VM.
    # For a 4 GB VM also (same rows, order, values and dtypes):
    #   * flight_key, the only unique string, is read first via COPY (densely allocated strings);
    #   * the other columns (m.* order, then the cutoff features) are streamed in groups of
    #     LOAD_GROUP columns, each with the same ORDER BY on the unique key and the same 200k-row
    #     chunks, so a chunk's Python row objects stay small; the float32 rule is decided per
    #     column and chunk exactly as before;
    #   * repeated strings/dates share one object per value, and the chunks are joined column by
    #     column (db_utils.concat_columns: pd.concat's dtypes without holding everything twice).
    mart_cols = [c for c in pd.read_sql("SELECT * FROM warehouse.mart_flight_features LIMIT 0", engine).columns
                 if c != "flight_key"]
    select = [f"m.{c}" for c in mart_cols] + [f"c.{c}" for c in cf.CUTOFF_FEATURES]
    columns = {"flight_key": copy_text_column(engine, f"SELECT m.flight_key {rows}")}
    canon = {}

    def reduced(chunk):
        for c in chunk.columns:
            if chunk[c].dtype == "float64":
                chunk[c] = chunk[c].astype("float32")
            elif chunk[c].dtype == object:
                chunk[c] = shared_objects(chunk[c], canon.setdefault(c, {}))
        return chunk

    for i in range(0, len(select), LOAD_GROUP):
        group = concat_columns(reduced(chunk) for chunk in read_sql_chunks(
            engine, f"SELECT {', '.join(select[i:i + LOAD_GROUP])} {rows}", chunksize=200_000))
        assert all(len(v) == len(columns["flight_key"]) for v in group.values()), "column groups must align"
        columns.update(group)
        release_memory()
    df = pd.DataFrame(columns, copy=False)
    del columns, group
    n_mart = pd.read_sql(f"SELECT count(*) FROM warehouse.mart_flight_features WHERE flight_date >= DATE '{first}'"
                         f"{extra}", engine).iloc[0, 0]
    assert len(df) == n_mart, f"cutoff features missing for {n_mart - len(df)} mart flights"
    df["month_idx"] = df["flight_year"].astype(int) * 12 + df["flight_month"].astype(int) - 1
    log(f"Loaded {len(df):,} completed flights (months from {first}) with {len(cf.CUTOFF_FEATURES)} cutoff features")
    return df


def month_label(idx):
    return datetime(idx // 12, idx % 12 + 1, 1).strftime("%b %Y")


def smoothed_rate(fit_df, apply_df, cols, prior):
    stats = fit_df.groupby(cols)["is_delayed_15"].agg(["sum", "count"])
    rate = (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)
    rate.name = "rate"
    return apply_df[cols].join(rate, on=cols)["rate"].fillna(prior).to_numpy()


def group_median(fit_df, apply_df, cols, value_col):
    med = fit_df.groupby(cols)[value_col].median()
    med.name = "med"
    return apply_df[cols].join(med, on=cols)["med"].to_numpy()


def history_columns(hist, target, relative=RELATIVE_RATES):
    """{history feature: values for the target rows}, computed from `hist` only."""
    out = {}
    prior = hist["is_delayed_15"].mean()
    for name, cols in RATE_GROUPS.items():
        rate = smoothed_rate(hist, target, cols, prior)
        out[name] = rate / prior if relative else rate
    for name, col in ROTATION_FEATURES.items():
        out[name] = group_median(hist, target, ROTATION_KEY, col)
    return out


def month_rows(df, months):
    """Row slice of consecutive `months` (df is ordered by date, so each month is one block)."""
    month_idx = df["month_idx"].to_numpy()
    assert np.all(np.diff(month_idx) >= 0) and list(months) == list(range(months[0], months[-1] + 1))
    lo, hi = np.searchsorted(month_idx, [months[0], months[-1] + 1])
    return slice(int(lo), int(hi))


def add_prev_month_history(df, months, relative=RELATIVE_RATES):
    """Fixed-length history: every row of month m gets features from month m-1 only.
    Rows of `months` in order plus the history columns: the same frame as concatenating one
    copied frame per month, built from views of each month (one copy instead of three)."""
    parts = {}
    for m in months:
        hist = df.iloc[month_rows(df, [m - 1])]
        assert len(hist) > 0, f"month {m - 1} is needed as history for month {m}"
        for name, values in history_columns(hist, df.iloc[month_rows(df, [m])], relative).items():
            parts.setdefault(name, []).append(values)
    rows = month_rows(df, months)
    # column-wise copies (reset_index / copy would also re-consolidate: a second full copy)
    out = pd.DataFrame({c: df[c].to_numpy()[rows].copy() for c in df.columns}, copy=False)
    for name, values in parts.items():
        out[name] = np.concatenate(values)
    return out


def map_numeric_columns(df):
    """The same frame with its numeric columns as read-only memory maps of temporary .npy files
    (np.save / np.load round-trip exactly). Memory only: mapped file pages are cache the kernel
    can drop and re-read under pressure, not anonymous memory, which leaves the model fits
    (~2 GB transient for the 1.1M-row refit: sklearn's float64 copies and median imputation)
    room in a 4 GB Docker VM. The files are removed at once; the mappings keep the data until
    the process ends. Nothing downstream writes into these frames."""
    columns = {}
    with tempfile.TemporaryDirectory(prefix="flight_train_") as tmp:
        for i, c in enumerate(df.columns):
            values = df[c].to_numpy()
            if values.dtype.kind in "biuf":
                path = os.path.join(tmp, f"{i}.npy")
                np.save(path, values)
                values = np.load(path, mmap_mode="r")
            columns[c] = values
    return pd.DataFrame(columns, copy=False)


# ---------------------------------------------------------------- models
def to_model_frame(df, carriers, features=FEATURES):
    # df[features] is already a new frame; a shallow copy only detaches it (a deep copy would
    # duplicate and re-consolidate it, ~0.6 GB on the refit data)
    X = df[features].copy(deep=False)
    X["carrier_code"] = pd.Categorical(X["carrier_code"], categories=carriers)
    return X


def fit_logreg(train, features, weighted=CLASS_WEIGHTED):
    """Score with logreg.predict_proba(frame with train's columns): the ColumnTransformer picks
    `features` by name (remainder dropped), so the frame is passed whole instead of as a
    train[features] copy. StandardScaler(copy=False) scales the imputer's fresh output in place
    (same arithmetic). Both are memory only: ~1 GB less at the peak of the 1.1M-row refit."""
    numeric = [f for f in features if f not in CATEGORICAL]
    logreg = make_pipeline(
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler(copy=False)), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]),
        LogisticRegression(class_weight="balanced" if weighted else None, max_iter=2000, random_state=SEED),
    )
    logreg.fit(train, train["is_delayed_15"].to_numpy())
    return logreg


def fit_lgbm(train, carriers, features, weighted=CLASS_WEIGHTED):
    y = train["is_delayed_15"].to_numpy()
    spw = (len(y) - y.sum()) / y.sum()
    gbm = lgb.LGBMClassifier(**LGB_PARAMS, scale_pos_weight=spw if weighted else 1.0)
    gbm.fit(to_model_frame(train, carriers, features), y, categorical_feature=CATEGORICAL)
    return gbm


def fit_models(train, carriers, features=FEATURES, include_logreg=True):
    """Returns {model name: scorer(df) -> score array}. The two rules need no fitting."""
    gbm = fit_lgbm(train, carriers, features)
    models = {
        RULE: lambda d: d["hist_rate_carrier_route"].to_numpy(),
        INB_RULE: lambda d: d["inb_known_delay_min"].to_numpy(dtype="float64"),
        LGBM: lambda d: gbm.predict_proba(to_model_frame(d, carriers, features))[:, 1],
        "_lgbm": gbm,
    }
    if include_logreg:
        logreg = fit_logreg(train, features)
        models[LOGREG] = lambda d: logreg.predict_proba(d)[:, 1]
    return models


def score_all(models, df, names=MODEL_NAMES):
    return {name: models[name](df) for name in names}


# ---------------------------------------------------------------- metrics
def top_share_flags(score, share=FLAG_SHARE):
    """Flags exactly the top `share` of rows by score (ties broken by row order)."""
    k = int(round(share * len(score)))
    flags = np.zeros(len(score), dtype=int)
    flags[np.argsort(-score, kind="stable")[:k]] = 1
    return flags


def at_flags(y, pred):
    return {"precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y, pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "predicted_positive_rate": round(float(np.mean(pred)), 4),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def evaluate(y, score):
    """Threshold-free AUCs + metrics at the 20% (top level) and 10% flag rates."""
    out = {"auc_roc": round(float(roc_auc_score(y, score)), 4),
           "auc_pr": round(float(average_precision_score(y, score)), 4)}
    out.update(at_flags(y, top_share_flags(score, FLAG_SHARE)))
    out["at_10pct"] = at_flags(y, top_share_flags(score, 0.10))
    return out


def choose_inbound_threshold(y, known):
    """Validation month: X (minutes) maximising F1 of 'flag if inbound known delay >= X'."""
    rows = []
    for x in INBOUND_RULE_CANDIDATES:
        pred = (known >= x).astype(int) if x > 0 else (known > 0).astype(int)
        rows.append({"threshold_min": x, **{k: v for k, v in at_flags(y, pred).items() if k != "confusion_matrix"}})
    best = max(rows, key=lambda r: (r["f1_score"], -r["threshold_min"]))
    return best["threshold_min"], rows


def inbound_rule_pred(known, x):
    return (known >= x).astype(int) if x > 0 else (known > 0).astype(int)


def calibration_table(y, prob, bins=10):
    q = pd.qcut(pd.Series(prob).rank(method="first"), bins, labels=False)
    t = pd.DataFrame({"bin": q, "p": prob, "y": y}).groupby("bin").agg(
        flights=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
    return [{"decile": int(i) + 1, "flights": int(r.flights), "mean_predicted": round(float(r.mean_predicted), 4),
             "actual_rate": round(float(r.actual_rate), 4)} for i, r in t.iterrows()]


def platt_diagnostic(y, score):
    z = np.log(np.clip(score, 1e-6, 1 - 1e-6) / (1 - np.clip(score, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return {"mean_score": round(float(np.mean(score)), 4), "platt_slope": round(float(lr.coef_[0, 0]), 3),
            "platt_intercept": round(float(lr.intercept_[0]), 3)}


def day_block_bootstrap(df, y, score_a, score_b, reps=200):
    """Paired bootstrap over DAYS: 95% interval of AUC-ROC / AUC-PR differences a - b."""
    rng = np.random.RandomState(SEED)
    days = df["flight_date"].to_numpy()
    uniq = np.unique(days)
    idx_by_day = [np.flatnonzero(days == d) for d in uniq]
    diffs = {"auc_roc": [], "auc_pr": []}
    for _ in range(reps):
        idx = np.concatenate([idx_by_day[i] for i in rng.randint(0, len(uniq), len(uniq))])
        yy = y[idx]
        diffs["auc_roc"].append(roc_auc_score(yy, score_a[idx]) - roc_auc_score(yy, score_b[idx]))
        diffs["auc_pr"].append(average_precision_score(yy, score_a[idx]) - average_precision_score(yy, score_b[idx]))
    out = {"reps": reps}
    for k, v in diffs.items():
        f = roc_auc_score if k == "auc_roc" else average_precision_score
        out[k] = {"difference": round(float(f(y, score_a) - f(y, score_b)), 4),
                  "ci95": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]}
    return out


# ---------------------------------------------------------------- outputs
def write_predictions(preds):
    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS warehouse.model_test_predictions")
        conn.exec_driver_sql("""
            CREATE TABLE warehouse.model_test_predictions (
                flight_key TEXT PRIMARY KEY, flight_date DATE, carrier_code TEXT, origin TEXT,
                dest TEXT, route TEXT, dep_hour SMALLINT, actual_delayed SMALLINT,
                delay_probability DOUBLE PRECISION, predicted_delayed SMALLINT)""")
        buf = io.StringIO()
        preds.to_csv(buf, index=False, header=False)
        buf.seek(0)
        conn.connection.cursor().copy_expert("COPY warehouse.model_test_predictions FROM STDIN WITH (FORMAT csv)", buf)


def reasons_for(contrib, values, n=3, batch=10_000):
    top = np.argsort(-np.abs(contrib), axis=1, kind="stable")[:, :n]
    # row dicts are built per batch: one dict per flagged flight (~115k x 53 values) needs ~0.5 GB
    records = (rec for i in range(0, len(values), batch)
               for rec in values.iloc[i:i + batch].to_dict(orient="records"))
    out = []
    for row, rec, c in zip(top, records, contrib):
        parts = []
        for i in row:
            feat = FEATURES[i]
            val = rec[feat]
            if pd.isna(val):
                shown = "unknown at cutoff" if feat.startswith(("inb_", "ac_")) else "n/a"
            elif feat.startswith("hist_rate") or isinstance(val, float) and not float(val).is_integer():
                shown = f"{val:.3f}" if feat.startswith("hist_rate") else f"{val:.1f}"
            else:
                shown = int(val) if isinstance(val, (float, np.floating)) else val
            parts.append(f"{READABLE[feat]}={shown} ({'+' if c[i] > 0 else '-'}{abs(c[i]):.2f})")
        out.append("; ".join(parts))
    return out


def fmt_row(name, m, w=52):
    t = m["at_10pct"]
    return (f"{name:{w}s}{m['auc_roc']:9.4f}{m['auc_pr']:9.4f}{m['precision']:8.4f}{m['recall']:8.4f}"
            f"{m['f1_score']:8.4f}{t['precision']:9.4f}{t['recall']:8.4f}\n")


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    months = sorted(int(m) for m in df["month_idx"].unique())
    assert months == list(range(months[0], months[0] + len(months))) and len(months) >= 4, \
        "need >= 4 consecutive months: history, fit, validation, test"
    history_month, test_month, val_month = months[0], months[-1], months[-2]
    stage1_months, final_months = months[1:-2], months[1:-1]
    carriers = sorted(df["carrier_code"].unique())

    feat = add_prev_month_history(df, months[1:])
    del df
    feat = map_numeric_columns(feat)
    release_memory()

    def in_months(ms):
        # A view of feat's rows (months are consecutive blocks), re-indexed from 0 like the former
        # boolean-mask copy; none of these frames is modified, so the four splits share one copy.
        part = feat.iloc[month_rows(feat, ms)]
        part.index = pd.RangeIndex(len(part))
        return part

    stage1, val, final_train, test = in_months(stage1_months), in_months([val_month]), \
        in_months(final_months), in_months([test_month])
    del feat
    assert stage1["flight_date"].max() < val["flight_date"].min() <= val["flight_date"].max() \
        < test["flight_date"].min(), "fit < validation < test in time"
    assert final_train["flight_date"].max() < test["flight_date"].min()
    y_val, y_final, y_test = (d["is_delayed_15"].to_numpy() for d in (val, final_train, test))
    L = month_label
    log(f"History-only {L(history_month)}; stage-1 fit {[L(m) for m in stage1_months]} ({len(stage1):,}); "
        f"validation {L(val_month)} ({len(val):,}, {y_val.mean():.2%} delayed); final fit "
        f"{[L(m) for m in final_months]} ({len(final_train):,}, {y_final.mean():.2%}); test {L(test_month)} "
        f"({len(test):,}, {y_test.mean():.2%})")

    # ---- Stage 1: fit on the earlier month(s), evaluate on validation
    models1 = fit_models(stage1, carriers)
    val_scores = score_all(models1, val)
    validation, val_calibration = {}, {}
    for n in MODEL_NAMES:
        s = val_scores[n]
        m = evaluate(y_val, s)
        validation[n] = m
        if n in (LOGREG, LGBM):
            val_calibration[n] = {"actual_rate": round(float(y_val.mean()), 4), **platt_diagnostic(y_val, s),
                                  "deciles": calibration_table(y_val, s)}
        log(f"[validation {L(val_month)}] {n}: AUC-ROC {m['auc_roc']} AUC-PR {m['auc_pr']} "
            f"P20 {m['precision']} R20 {m['recall']} P10 {m['at_10pct']['precision']} R10 {m['at_10pct']['recall']}")
    inb_x, inb_sweep = choose_inbound_threshold(y_val, val["inb_known_delay_min"].to_numpy(dtype="float64"))
    inb_val = at_flags(y_val, inbound_rule_pred(val["inb_known_delay_min"].to_numpy(dtype="float64"), inb_x))
    log(f"Inbound-delay rule threshold chosen on validation (best F1): >= {inb_x} min -> {inb_val}")
    del models1, val_scores
    release_memory()

    # ---- Stage 2: refit on all pre-test months, score the test month ONCE
    models = fit_models(final_train, carriers)
    model = models["_lgbm"]
    test_scores = score_all(models, test)
    results, calibration = {}, {}
    for n in MODEL_NAMES:
        s = test_scores[n]
        results[n] = evaluate(y_test, s)
        if n in (LOGREG, LGBM):
            calibration[n] = {
                "actual_rate": round(float(y_test.mean()), 4),
                "mean_predicted_probability": round(float(s.mean()), 4),
                "brier": round(float(brier_score_loss(y_test, s)), 5),
                "brier_constant_train_rate": round(float(brier_score_loss(
                    y_test, np.full(len(y_test), y_final.mean()))), 5),
                "deciles": calibration_table(y_test, s),
            }
        log(f"[test {L(test_month)}] {n}: AUC-ROC {results[n]['auc_roc']} AUC-PR {results[n]['auc_pr']} "
            f"P20 {results[n]['precision']} R20 {results[n]['recall']} "
            f"P10 {results[n]['at_10pct']['precision']} R10 {results[n]['at_10pct']['recall']}")
    known_test = test["inb_known_delay_min"].to_numpy(dtype="float64")
    inb_rule_test = {"threshold_min_chosen_on_validation": inb_x, "validation": inb_val,
                     "test": at_flags(y_test, inbound_rule_pred(known_test, inb_x))}
    log(f"[test] inbound-delay rule at its validation threshold >= {inb_x} min: {inb_rule_test['test']}")
    if results[LGBM]["auc_roc"] > 0.92:
        print("WARNING: AUC-ROC > 0.92 two hours before departure - investigate leakage before reporting.")
    auc_diff = day_block_bootstrap(test, y_test, test_scores[LGBM], test_scores[LOGREG])
    log(f"LightGBM - logistic regression, day-block bootstrap: {auc_diff}")

    # Reference: the old weeks-ahead feature set, same split and settings (like-for-like on this test month)
    ref_gbm = fit_lgbm(final_train, carriers, WEEKS_AHEAD_FEATURES)
    ref_score = ref_gbm.predict_proba(to_model_frame(test, carriers, WEEKS_AHEAD_FEATURES))[:, 1]
    reference = {REF_LGBM: evaluate(y_test, ref_score)}
    log(f"[test] {REF_LGBM}: AUC-ROC {reference[REF_LGBM]['auc_roc']} AUC-PR {reference[REF_LGBM]['auc_pr']}")
    del ref_gbm
    release_memory()

    # Robustness: flights whose inbound rotation looks as planned (no same-day swap signature)
    turn = test["inb_sched_turn_min"].to_numpy(dtype="float64")
    swap = ~np.isnan(turn) & (turn < SWAP_SIGNATURE_TURN_MIN)
    keep = ~swap
    robustness = {
        "definition": (f"test flights excluding those whose scheduled turn after the inbound leg is < "
                       f"{SWAP_SIGNATURE_TURN_MIN} min or negative (the signature of a same-day aircraft swap, "
                       "which BTS tail numbers record after the fact)"),
        "excluded_flights": int(swap.sum()), "excluded_delay_rate": round(float(y_test[swap].mean()), 4)
        if swap.any() else None,
        "kept_flights": int(keep.sum()), "kept_delay_rate": round(float(y_test[keep].mean()), 4),
        "models": {n: {"auc_roc": round(float(roc_auc_score(y_test[keep], test_scores[n][keep])), 4),
                       "auc_pr": round(float(average_precision_score(y_test[keep], test_scores[n][keep])), 4)}
                   for n in MODEL_NAMES},
    }
    log(f"Robustness (no swap signature): {robustness}")

    raw = test_scores[LGBM]
    pred = top_share_flags(raw)

    # SHAP (exact TreeSHAP via LightGBM pred_contrib, log-odds)
    X_test = to_model_frame(test, carriers)

    def shap_contrib(X):
        return model.booster_.predict(X, pred_contrib=True, num_threads=8)[:, :-1]

    sample_idx = np.sort(np.random.RandomState(SEED).choice(len(X_test), size=min(20000, len(X_test)),
                                                            replace=False))
    contrib_sample = shap_contrib(X_test.iloc[sample_idx])
    mean_abs = np.abs(contrib_sample).mean(axis=0)
    importance = (pd.DataFrame({"feature": FEATURES, "readable_name": [READABLE[f] for f in FEATURES],
                                "mean_abs_shap": np.round(mean_abs, 5)})
                  .sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True))
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    importance["group"] = importance["feature"].map(lambda f: group_of.get(f, "schedule_history"))
    group_share = (importance.groupby("group")["mean_abs_shap"].sum() / importance["mean_abs_shap"].sum()).round(4)
    print(importance.head(15).to_string(index=False))

    import shap
    shap_sample = X_test.iloc[sample_idx].copy()
    shap_sample["carrier_code"] = shap_sample["carrier_code"].cat.codes
    plt.figure()
    shap.summary_plot(contrib_sample, shap_sample, feature_names=[READABLE[f] for f in FEATURES],
                      show=False, max_display=15)
    plt.title(f"SHAP, 20k {L(test_month)} flights, 2 h before departure: push toward 'delayed 15+ min'",
              fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=130)
    plt.close()

    preds = test[["flight_key", "flight_date", "carrier_code", "origin", "dest", "route", "dep_hour"]].copy()
    preds["actual_delayed"] = y_test
    preds["delay_probability"] = np.round(raw, 5)
    preds["predicted_delayed"] = pred
    write_predictions(preds)
    log(f"Wrote {len(preds):,} rows to warehouse.model_test_predictions")

    flagged_idx = np.flatnonzero(pred == 1)
    flagged = preds.iloc[flagged_idx].copy()
    values = test[FEATURES]
    flagged["inbound_known_delay_min"] = test["inb_known_delay_min"].to_numpy()[flagged_idx]
    flagged["top_reasons"] = reasons_for(shap_contrib(X_test.iloc[flagged_idx]), values.iloc[flagged_idx])
    flagged = flagged.sort_values(["delay_probability", "flight_key"], ascending=[False, True])
    cm = np.array(results[LGBM]["confusion_matrix"])
    assert len(flagged) == cm[0, 1] + cm[1, 1], "flagged list must equal predicted positives"
    flagged.to_csv(OUTPUTS_DIR / "predicted_delays_test_month.csv", index=False)
    log(f"Wrote {len(flagged):,} flagged flights (= TP {cm[1, 1]:,} + FP {cm[0, 1]:,})")

    ex_idx = list(np.argsort(-raw, kind="stable")[:3]) + list(np.argsort(raw, kind="stable")[:2])
    examples = preds.iloc[ex_idx].copy()
    examples["top_reasons"] = reasons_for(shap_contrib(X_test.iloc[ex_idx]), values.iloc[ex_idx])
    examples.to_csv(OUTPUTS_DIR / "example_predictions.csv", index=False)

    model.booster_.save_model(str(MODEL_DIR / "lgbm_delay_model.txt"))
    meta = {"trained_months": [L(m) for m in final_months], "test_month": L(test_month),
            "features": FEATURES}
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    metrics = {
        "question": "Will this flight arrive 15+ minutes late? Predicted 2 hours before scheduled departure.",
        "label": "ArrDel15 (arrived 15+ minutes late), completed flights only",
        "prediction_point": "scheduled departure - 2 h (UTC, time-zone aware); only events before it are used",
        "split": "time-based over the last 4 months: history / fit / validation / test",
        "history_features": "previous calendar month for every row (fixed-length window); rates as lift",
        "history_only_months": [L(history_month)],
        "stage1_fit_months": [L(m) for m in stage1_months],
        "validation_month": L(val_month),
        "train_months": [L(m) for m in final_months],
        "test_month": L(test_month),
        "test_month_start": f"{test_month // 12}-{test_month % 12 + 1:02d}-01",
        "stage1_fit_size": int(len(stage1)), "validation_size": int(len(val)),
        "train_size": int(len(final_train)), "test_size": int(len(test)),
        "validation_delay_rate": round(float(y_val.mean()), 4),
        "train_delay_rate": round(float(y_final.mean()), 4),
        "test_delay_rate": round(float(y_test.mean()), 4),
        "operating_point": ("each model flags the riskiest 20% (and, separately, 10%) of the scored month's "
                            "flights; precision/recall/F1 at the top level are the 20% values"),
        "flag_share": FLAG_SHARE,
        "class_weighted": CLASS_WEIGHTED,
        "relative_rates": RELATIVE_RATES,
        "feature_groups_used": USED_GROUPS,
        "features": FEATURES,
        "excluded_same_day_tail_schedule_features": EXCLUDED_TAIL_FEATURES,
        "validation_models": validation,
        "validation_calibration": val_calibration,
        "inbound_rule": inb_rule_test,
        "inbound_rule_threshold_sweep_validation": inb_sweep,
        "models": results,
        "reference_models": reference,
        "robustness_no_swap_signature": robustness,
        "calibration_check_test": calibration,
        "lightgbm_minus_logreg_day_bootstrap": auc_diff,
        "main_model": LGBM,
        "top_features": importance.head(15).to_dict(orient="records"),
        "shap_share_by_group": group_share.to_dict(),
        "flagged_rows_written": int(len(flagged)),
    }
    (OUTPUTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    importance.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    with open(OUTPUTS_DIR / "model_report.txt", "w") as f:
        f.write("Flight Arrival Delay (15+ min) Model Report - predicted 2 hours before scheduled departure\n"
                + "=" * 88 + "\n\n")
        f.write(f"History-only month: {L(history_month)} (previous-calendar-month rate features)\n")
        f.write(f"Stage 1: fit on {[L(m) for m in stage1_months]} ({len(stage1):,} flights), validation "
                f"{L(val_month)} ({len(val):,} flights, {y_val.mean():.2%} delayed)\n")
        f.write(f"Final:   refit on {[L(m) for m in final_months]} ({len(final_train):,} flights, "
                f"{y_final.mean():.2%} delayed), test {L(test_month)} ({len(test):,} flights, "
                f"{y_test.mean():.2%} delayed), scored once\n\n")
        hdr = (f"{'Model':52s}{'AUC-ROC':>9s}{'AUC-PR':>9s}{'P@20%':>8s}{'R@20%':>8s}{'F1@20%':>8s}"
               f"{'P@10%':>9s}{'R@10%':>8s}\n")
        for title, block in (("VALIDATION month (stage-1 models)", validation),
                             ("TEST month (refit models)", {**results, **reference})):
            f.write(f"{title}\n{hdr}")
            for name, m in block.items():
                f.write(fmt_row(name, m))
            f.write("\n")
        f.write(f"Inbound-delay rule as a yes/no rule: flag if inbound known delay >= {inb_x} min (best F1 on "
                f"validation). Test: {inb_rule_test['test']}\n")
        f.write(f"AUC-PR of a random classifier = test delay rate = {y_test.mean():.4f}\n")
        f.write(f"LightGBM minus logistic regression (paired day-block bootstrap, {auc_diff['reps']} reps): "
                f"AUC-ROC {auc_diff['auc_roc']['difference']:+.4f} (95% CI {auc_diff['auc_roc']['ci95']}), "
                f"AUC-PR {auc_diff['auc_pr']['difference']:+.4f} (95% CI {auc_diff['auc_pr']['ci95']})\n")
        f.write(f"\nRobustness: {robustness}\n")
        f.write("\nCalibration, test month (probabilities as published):\n")
        for name, c in calibration.items():
            f.write(f"  {name}: actual {c['actual_rate']:.4f} | mean predicted {c['mean_predicted_probability']:.4f} | "
                    f"Brier {c['brier']:.5f} (constant train rate {c['brier_constant_train_rate']:.5f})\n")
        f.write(f"\nLightGBM confusion matrix at 20% [rows=actual 0/1, cols=predicted 0/1]:\n{cm}\n")
        f.write(f"\nSHAP share by feature group: {group_share.to_dict()}\n")
        f.write(f"\nTop features by mean |SHAP|:\n{importance.head(15).to_string(index=False)}\n")
        f.write("\nExample predictions (p = predicted probability; reasons = SHAP, log-odds):\n")
        for _, r in examples.iterrows():
            f.write(f"  {r['flight_date']} {r['carrier_code']} {r['route']} dep hour {r['dep_hour']}: "
                    f"p={r['delay_probability']:.3f}, actual={'delayed' if r['actual_delayed'] else 'on time'}"
                    f"\n    {r['top_reasons']}\n")

    print("\n=== TEST-MONTH METRICS ===")
    for name, m in {**results, **reference}.items():
        print(f"{name:52s} AUC-ROC={m['auc_roc']} AUC-PR={m['auc_pr']} P20={m['precision']} R20={m['recall']} "
              f"P10={m['at_10pct']['precision']} R10={m['at_10pct']['recall']}")


if __name__ == "__main__":
    main()
