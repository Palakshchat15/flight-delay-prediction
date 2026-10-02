"""
Exports the small, dashboard-ready tables that the Excel and Tableau builders read
into data/processed/ (from the dbt marts and the model outputs).

Rates are written as percentages (0-100, one decimal) so axes read naturally in
both tools; the underlying counts are included so KPIs can be recomputed.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ml_pipeline"))
from db_utils import get_engine  # noqa: E402

PROCESSED_DIR = Path(os.environ.get("DASHBOARD_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

METRIC_COLS = ("scheduled_flights, cancelled_flights, diverted_flights, completed_flights, delayed_flights, "
               "round(100 * on_time_pct, 1) as on_time_pct, round(100 * delay_rate, 1) as delay_rate_pct, "
               "round(100 * cancellation_rate, 2) as cancellation_rate_pct, avg_arr_delay_minutes")

QUERIES = {
    "ontime_by_carrier": f"""
        SELECT carrier_name, carrier_code, {METRIC_COLS}
        FROM warehouse.mart_ontime_by_carrier ORDER BY on_time_pct DESC, carrier_code""",
    # Airports with at least 1,000 completed flights in the period (~busiest 150); map + ranking.
    "ontime_by_origin": f"""
        SELECT airport_code, airport_name, city_name, state_code, latitude, longitude, {METRIC_COLS}
        FROM warehouse.mart_ontime_by_origin
        WHERE completed_flights >= 1000 AND latitude IS NOT NULL
        ORDER BY scheduled_flights DESC, airport_code""",
    "ontime_by_hour": f"""
        SELECT dep_hour, {METRIC_COLS}
        FROM warehouse.mart_ontime_by_hour ORDER BY dep_hour""",
    "ontime_by_dow_month": f"""
        SELECT flight_month, month_name, day_of_week, day_name, {METRIC_COLS}
        FROM warehouse.mart_ontime_by_dow_month ORDER BY flight_month, day_of_week""",
    "ontime_daily": f"""
        SELECT flight_date, {METRIC_COLS}
        FROM warehouse.mart_ontime_daily ORDER BY flight_date""",
    "top_delayed_routes": f"""
        SELECT delay_rank, route, origin_airport_code, dest_airport_code, {METRIC_COLS}
        FROM warehouse.mart_top_delayed_routes WHERE delay_rank <= 20 ORDER BY delay_rank""",
    "delay_causes_by_month": """
        SELECT flight_month, month_name, cause, delay_minutes,
               round(100.0 * delay_minutes / sum(delay_minutes) over (partition by flight_month), 1)
                   AS share_of_month_pct
        FROM warehouse.mart_delay_causes_by_month ORDER BY flight_month, cause""",
    # Carrier x scheduled hour, counts only: Tableau computes the rate as SUM/SUM so any filter
    # selection (one carrier or several) stays a correct weighted rate.
    "delay_by_carrier_hour": """
        SELECT c.carrier_name, f.dep_hour,
               count(*) AS completed_flights, sum(f.is_delayed_15) AS delayed_flights
        FROM warehouse.fact_flights f JOIN warehouse.dim_carrier c USING (carrier_code)
        WHERE NOT f.is_cancelled AND NOT f.is_diverted
        GROUP BY 1, 2 ORDER BY 1, 2""",
    # Test month only: actual delay rate vs the model's mean predicted probability (unweighted
    # LightGBM, so a real probability), by carrier. flagged_pct = share in the riskiest 20%.
    "model_by_carrier": """
        SELECT c.carrier_name, count(*) AS test_flights,
               round(100 * avg(p.actual_delayed), 1) AS actual_delay_rate_pct,
               round(100 * avg(p.delay_probability)::numeric, 1) AS mean_predicted_probability_pct,
               round(100 * avg(p.predicted_delayed), 1) AS flagged_pct
        FROM warehouse.model_test_predictions p JOIN warehouse.dim_carrier c USING (carrier_code)
        GROUP BY 1 ORDER BY actual_delay_rate_pct DESC, 1""",
}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    frames = {}
    for name, sql in QUERIES.items():
        frames[name] = pd.read_sql(sql, engine)

    # Long format of model_by_carrier for Tableau's colour-by-series dot plot.
    mbc = frames["model_by_carrier"]
    frames["model_by_carrier_long"] = pd.concat([
        mbc[["carrier_name"]].assign(series="Actual delay rate", rate_pct=mbc["actual_delay_rate_pct"]),
        mbc[["carrier_name"]].assign(series="Mean predicted probability",
                                     rate_pct=mbc["mean_predicted_probability_pct"]),
    ], ignore_index=True)

    metrics = json.loads((OUTPUTS_DIR / "model_metrics.json").read_text())
    rows = []
    for model, m in metrics["models"].items():
        rows.append({"model": model, "auc_pr": m["auc_pr"], "auc_roc": m["auc_roc"],
                     "precision": m["precision"], "recall": m["recall"], "f1_score": m["f1_score"],
                     "accuracy": m["accuracy"], "predicted_positive_rate": m["predicted_positive_rate"],
                     "precision_at_10pct": m["at_10pct"]["precision"], "recall_at_10pct": m["at_10pct"]["recall"],
                     "test_month": metrics["test_month"]})
    frames["model_metrics"] = pd.DataFrame(rows)
    # Like-for-like on the same test month: LightGBM with the old weeks-ahead feature set (refit in
    # the same run) vs the 2-hours-before model.
    ref = metrics["reference_models"]["LightGBM, weeks-ahead features only (reference)"]
    new_m = metrics["models"]["LightGBM"]
    frames["model_before_after"] = pd.DataFrame([
        {"feature_set": fs, "auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"], "precision": m["precision"],
         "recall": m["recall"], "precision_at_10pct": m["at_10pct"]["precision"],
         "recall_at_10pct": m["at_10pct"]["recall"]}
        for fs, m in (("Weeks ahead (schedule + history)", ref), ("2 hours before departure", new_m))])
    labels = {"auc_pr": "AUC-PR", "auc_roc": "AUC-ROC", "precision": "Precision",
              "recall": "Recall", "f1_score": "F1"}
    frames["model_metrics_long"] = (frames["model_metrics"]
                                    .melt(id_vars="model", value_vars=list(labels), var_name="metric")
                                    .assign(metric=lambda d: d["metric"].map(labels)))
    cm = metrics["models"][metrics["main_model"]]["confusion_matrix"]
    frames["confusion_matrix"] = pd.DataFrame([
        {"actual": "On time", "predicted": "On time", "flights": cm[0][0]},
        {"actual": "On time", "predicted": "Delayed", "flights": cm[0][1]},
        {"actual": "Delayed", "predicted": "On time", "flights": cm[1][0]},
        {"actual": "Delayed", "predicted": "Delayed", "flights": cm[1][1]},
    ])
    fi = pd.read_csv(OUTPUTS_DIR / "feature_importance.csv")
    frames["feature_importance"] = fi[["readable_name", "mean_abs_shap"]].rename(
        columns={"readable_name": "feature"})

    # Cross-checks between outputs before anything is published.
    n_test = metrics["test_size"]
    assert int(mbc["test_flights"].sum()) == n_test, "model_by_carrier must cover every test flight"
    assert int(frames["confusion_matrix"]["flights"].sum()) == n_test, "confusion matrix must sum to test size"
    flagged = pd.read_csv(OUTPUTS_DIR / "predicted_delays_test_month.csv", usecols=["flight_key"])
    assert len(flagged) == cm[0][1] + cm[1][1], "flagged list must equal predicted positives"
    total = frames["ontime_by_carrier"]["scheduled_flights"].sum()
    assert total == frames["ontime_by_hour"]["scheduled_flights"].sum() == \
        frames["ontime_daily"]["scheduled_flights"].sum(), "aggregate marts must agree on total flights"

    for name, df in frames.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} rows")
    print(f"Cross-checks passed: {total:,} scheduled flights; {n_test:,} test flights; "
          f"{len(flagged):,} flagged = predicted positives")


if __name__ == "__main__":
    main()
