"""
Design checks for the 2-hours-before-departure model, on the VALIDATION month only
(the test month is excluded in the SQL and never read).

Every variant fits on the stage-1 month and is scored on the validation month with the
same models and seeds as train_delay_model.py:
  * feature-group ablation: weeks-ahead features (schedule + previous-month history) ->
    + inbound aircraft -> + congestion -> + origin weather -> + destination weather
    (forecast stand-in). A group is kept only if it does not hurt validation AUC-PR.
  * inbound-delay rule threshold sweep (X minutes), for the rule baseline.
The earlier weeks-ahead checks (history scheme, class weighting, lift) are kept in
outputs/design_checks_weeks_ahead.json and their choices are reused unchanged.

Writes outputs/design_checks.json. Not part of the DAG (about 15 minutes); run with
  docker compose exec airflow-scheduler python /opt/airflow/src/ml_pipeline/design_checks.py
"""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import train_delay_model as t


def run_variant(name, fit, val, carriers, features):
    y = val["is_delayed_15"].to_numpy()
    out = {"variant": name, "n_features": len(features)}
    gbm = t.fit_lgbm(fit, carriers, features)
    lr = t.fit_logreg(fit, features)
    for n, s in ((t.LOGREG, lr.predict_proba(val)[:, 1]),
                 (t.LGBM, gbm.predict_proba(t.to_model_frame(val, carriers, features))[:, 1])):
        m = t.evaluate(y, s)
        out[n] = {"auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"], "precision_20": m["precision"],
                  "recall_20": m["recall"], "precision_10": m["at_10pct"]["precision"],
                  "recall_10": m["at_10pct"]["recall"], "mean_score": round(float(np.mean(s)), 4)}
    t.log(f"{name}: {out}")
    del gbm, lr
    import gc
    gc.collect()
    return out


def main():
    df = t.load_data(exclude_last_month=True)
    months = sorted(int(m) for m in df["month_idx"].unique())
    val_month, fit_months = months[-1], months[1:-1]
    carriers = sorted(df["carrier_code"].unique())
    feat = t.add_prev_month_history(df, months[1:])
    del df
    fit = feat[feat["month_idx"].isin(fit_months)].reset_index(drop=True)
    val = feat[feat["month_idx"] == val_month].reset_index(drop=True)
    del feat
    import gc
    gc.collect()
    y = val["is_delayed_15"].to_numpy()
    G = t.FEATURE_GROUPS
    base = t.WEEKS_AHEAD_FEATURES
    variants = [
        ("weeks-ahead features only (schedule + previous-month history)", base),
        ("+ inbound aircraft", base + G["inbound"]),
        ("+ inbound + congestion", base + G["inbound"] + G["congestion"]),
        ("+ inbound + congestion + origin weather", base + G["inbound"] + G["congestion"] + G["weather_origin"]),
        ("+ inbound + congestion + origin weather + destination weather (forecast stand-in) = ALL",
         base + G["inbound"] + G["congestion"] + G["weather_origin"] + G["weather_dest_forecast_standin"]),
        ("weeks-ahead + congestion + weather, NO inbound aircraft",
         base + G["congestion"] + G["weather_origin"] + G["weather_dest_forecast_standin"]),
    ]
    results = [run_variant(n, fit, val, carriers, f) for n, f in variants]
    known = val["inb_known_delay_min"].to_numpy(dtype="float64")
    x, sweep = t.choose_inbound_threshold(y, known)
    rule_scores = {"auc_roc": round(float(roc_auc_score(y, known)), 4),
                   "auc_pr": round(float(average_precision_score(y, known)), 4)}
    out = {"fit_months": [t.month_label(m) for m in fit_months], "validation_month": t.month_label(val_month),
           "test_month_used": False, "results": results,
           "inbound_rule": {"chosen_threshold_min": x, "criterion": "best F1 on validation",
                            "score_auc": rule_scores, "sweep": sweep}}
    (t.OUTPUTS_DIR / "design_checks.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'variant':95s}{'LR AUC':>8s}{'LR PR':>8s}{'GBM AUC':>9s}{'GBM PR':>8s}")
    for r in results:
        print(f"{r['variant']:95s}{r[t.LOGREG]['auc_roc']:8.4f}{r[t.LOGREG]['auc_pr']:8.4f}"
              f"{r[t.LGBM]['auc_roc']:9.4f}{r[t.LGBM]['auc_pr']:8.4f}")
    print(f"Inbound-delay rule: AUC {rule_scores}; best-F1 threshold >= {x} min")


if __name__ == "__main__":
    main()
