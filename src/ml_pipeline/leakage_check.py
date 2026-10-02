"""
Automated leakage check for the 2-hours-before-departure features (DAG task leakage_check).
Fails (exit 1) on any violation, which blocks dbt test and training.

1. Timestamp rule: every event time a feature used (*_ts in warehouse.flight_cutoff_features)
   is strictly before the flight's cutoff_utc (= scheduled departure - 2 h); origin weather
   hour <= cutoff. Every "known only after the event" value is NULL exactly when its event
   time is NULL (i.e. not yet happened at the cutoff).
2. Key cross-check in SQL against intermediate.int_flights_enriched: the inbound leg's
   departure / arrival delay is present only if that leg's actual departure / arrival was
   before the cutoff, and equals the recorded delay; the aircraft's "latest known" legs
   happened before the cutoff.
3. Independent recomputation: for a seeded random sample of flights, every inbound and
   congestion feature is recomputed by a deliberately naive per-flight implementation that
   first TRUNCATES the world to what was visible at the cutoff (other flights' departures
   with actual_dep < cutoff, arrivals with actual_arr < cutoff, plus the schedule), and must
   match the vectorised values in the table.
4. Feature-name rule: the model feature list contains none of the flight's own post-departure
   columns (dep delay, taxi, wheels, actual times, delay causes, cancellation/diversion).
5. Time-zone sanity: UTC scheduled block time (sched_arr_utc - sched_dep_utc) matches BTS's
   scheduled elapsed time within 5 min for >= 99% of flights.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cutoff_features as cf  # noqa: E402
from db_utils import get_engine  # noqa: E402

SAMPLE = int(os.environ.get("LEAKAGE_SAMPLE", "400"))
SEED = 42
FORBIDDEN_OWN_COLUMNS = {
    "dep_time", "dep_delay", "dep_delay_minutes", "dep_del15", "departure_delay_groups", "taxi_out", "taxi_in",
    "wheels_off", "wheels_on", "arr_time", "arr_delay", "arr_delay_minutes", "arr_del15", "actual_elapsed_time",
    "air_time", "carrier_delay_minutes", "weather_delay_minutes", "nas_delay_minutes", "security_delay_minutes",
    "late_aircraft_delay_minutes", "is_cancelled", "is_diverted", "cancellation_code", "cancellation_reason",
    "actual_dep_utc", "actual_arr_utc", "flight_status", "is_delayed_15",
}
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", Path(__file__).resolve().parents[2] / "outputs"))


def check_timestamps(engine, failures, report):
    conds = " + ".join(f"(({c} IS NOT NULL AND {c} >= cutoff_utc))::int" for c in cf.STRICT_TS)
    pairs = [("inb_prev_dep_delay", "inb_prev_dep_ts"), ("inb_prev_arr_delay", "inb_prev_arr_ts"),
             ("ac_latest_dep_delay", "ac_latest_dep_ts"), ("ac_latest_arr_delay", "ac_latest_arr_ts"),
             ("origin_recent_dep_delay_avg", "origin_dep_win_max_ts"),
             ("origin_recent_arr_delay_avg", "origin_arr_win_max_ts"),
             ("dest_recent_arr_delay_avg", "dest_arr_win_max_ts"), ("dest_recent_dep_delay_avg", "dest_dep_win_max_ts"),
             ("carrier_recent_dep_delay_avg", "carrier_win_max_ts"), ("nat_recent_dep_delay_avg", "nat_win_max_ts")]
    null_mismatch = " + ".join(f"(({v} IS NULL) <> ({t} IS NULL))::int" for v, t in pairs)
    row = pd.read_sql(f"""
        SELECT count(*) AS rows,
               sum({conds}) AS ts_at_or_after_cutoff,
               sum((wx_o_ts > cutoff_utc)::int) AS origin_weather_after_cutoff,
               sum({null_mismatch}) AS value_ts_null_mismatch,
               sum((inb_prev_dep_ts IS NOT NULL)::int) AS rows_with_prev_dep_known,
               sum((inb_prev_arr_ts IS NOT NULL)::int) AS rows_with_prev_arr_known,
               max(extract(epoch FROM (inb_prev_dep_ts - cutoff_utc)) / 60) AS max_prev_dep_minus_cutoff_min,
               max(extract(epoch FROM (nat_win_max_ts - cutoff_utc)) / 60) AS max_window_event_minus_cutoff_min,
               sum((wx_d_ts > cutoff_utc)::int) AS dest_weather_after_cutoff_forecast_standin
        FROM warehouse.flight_cutoff_features""", engine).iloc[0]
    report["timestamp_rule"] = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
    print("1. timestamp rule:", report["timestamp_rule"], flush=True)
    if row["ts_at_or_after_cutoff"] or row["origin_weather_after_cutoff"] or row["value_ts_null_mismatch"]:
        failures.append("timestamp rule violated")


def check_keys(engine, failures, report):
    # Four joins over ~2.2M rows: parallel hash joins exhaust Docker's /dev/shm, so run serially.
    with engine.connect() as conn:
        conn.exec_driver_sql("SET max_parallel_workers_per_gather = 0")
        row = _key_query(conn)
    report["key_cross_check"] = {k: int(v) for k, v in row.items()}
    print("2. key cross-check:", report["key_cross_check"], flush=True)
    if any(int(v) for v in row.values):
        failures.append("key cross-check violated")


def _key_query(conn):
    return pd.read_sql("""
        SELECT
          sum(((f.inb_prev_dep_delay IS NOT NULL) <> coalesce(p.actual_dep_utc < f.cutoff_utc, false))::int)
              AS prev_dep_known_mismatch,
          sum((f.inb_prev_dep_delay IS NOT NULL AND f.inb_prev_dep_delay <> p.dep_delay_minutes)::int)
              AS prev_dep_value_mismatch,
          sum(((f.inb_prev_arr_delay IS NOT NULL) <> coalesce(p.actual_arr_utc < f.cutoff_utc, false))::int)
              AS prev_arr_known_mismatch,
          sum((f.inb_prev_arr_delay IS NOT NULL AND f.inb_prev_arr_delay <> p.arr_delay_minutes)::int)
              AS prev_arr_value_mismatch,
          sum((p.flight_key IS NOT NULL AND p.sched_dep_utc >= s.sched_dep_utc)::int) AS prev_not_earlier,
          sum((d.flight_key IS NOT NULL AND NOT (d.actual_dep_utc < f.cutoff_utc))::int) AS latest_dep_not_before_cutoff,
          sum((a.flight_key IS NOT NULL AND NOT (a.actual_arr_utc < f.cutoff_utc))::int) AS latest_arr_not_before_cutoff,
          sum((f.prev_flight_key = f.flight_key OR f.ac_latest_dep_key = f.flight_key
               OR f.ac_latest_arr_key = f.flight_key)::int) AS own_flight_used,
          sum((f.prev_flight_key IS NOT NULL AND p.tail_number IS DISTINCT FROM s.tail_number)::int) AS prev_other_tail
        FROM warehouse.flight_cutoff_features f
        JOIN intermediate.int_flights_enriched s ON s.flight_key = f.flight_key
        LEFT JOIN intermediate.int_flights_enriched p ON p.flight_key = f.prev_flight_key
        LEFT JOIN intermediate.int_flights_enriched d ON d.flight_key = f.ac_latest_dep_key
        LEFT JOIN intermediate.int_flights_enriched a ON a.flight_key = f.ac_latest_arr_key""", conn).iloc[0]


def naive_features(i, ev, by_tail, by_origin_dep, by_dest_arr, by_carrier):
    """Deliberately simple per-flight recomputation from the world truncated at the cutoff."""
    r = ev.iloc[i]
    c = r["cutoff_utc_m"]
    out = {}
    # visible world: departures before the cutoff, arrivals before the cutoff
    dep_seen = lambda d: d[d["actual_dep_utc_m"] < c]  # noqa: E731
    arr_seen = lambda d: d[d["actual_arr_utc_m"] < c]  # noqa: E731

    # inbound: previous leg of the same tail = latest STRICTLY earlier scheduled departure, within 24 h
    prev = None
    if isinstance(r["tail_number"], str) and r["tail_number"] in by_tail.idx:
        legs = by_tail[r["tail_number"]]
        earlier = legs[((legs["sched_dep_utc_m"] < r["sched_dep_utc_m"]) |
                        False) & (legs["sched_dep_utc_m"] >= r["sched_dep_utc_m"] - cf.LOOKBACK_MIN)]
        if len(earlier):
            prev = earlier.sort_values(["sched_dep_utc_m", "row"]).iloc[-1]
        seen_d = dep_seen(legs)
        seen_d = seen_d[seen_d["actual_dep_utc_m"] >= c - cf.LOOKBACK_MIN]
        seen_a = arr_seen(legs)
        seen_a = seen_a[seen_a["actual_arr_utc_m"] >= c - cf.LOOKBACK_MIN]
        out["ac_latest_dep_delay"] = (seen_d.sort_values(["actual_dep_utc_m", "row"]).iloc[-1]["dep_delay_minutes"]
                                      if len(seen_d) else np.nan)
        out["ac_latest_arr_delay"] = (seen_a.sort_values(["actual_arr_utc_m", "row"]).iloc[-1]["arr_delay_minutes"]
                                      if len(seen_a) else np.nan)
    else:
        out["ac_latest_dep_delay"] = out["ac_latest_arr_delay"] = np.nan
    if prev is None:
        out.update(inb_has_prev=0, inb_prev_departed=0, inb_prev_arrived=0, inb_prev_sched_dep_before_cutoff=0,
                   inb_prev_dep_delay=np.nan, inb_prev_arr_delay=np.nan, inb_prev_overdue_min=0.0,
                   inb_known_delay_min=0.0, inb_sched_turn_min=np.nan, inb_projected_slack_min=np.nan)
    else:
        departed = bool(prev["actual_dep_utc_m"] < c)       # NaN (never departed) -> False
        arrived = bool(prev["actual_arr_utc_m"] < c)
        sched_before = bool(prev["sched_dep_utc_m"] < c)
        overdue = c - prev["sched_dep_utc_m"] if (sched_before and not departed) else 0.0
        if arrived:
            known = prev["arr_delay_minutes"]
        elif departed:
            known = max(prev["dep_delay_minutes"], max(c - prev["sched_arr_utc_m"], 0))
        elif sched_before:
            known = overdue
        else:
            known = 0.0
        turn = r["sched_dep_utc_m"] - prev["sched_arr_utc_m"]
        out.update(inb_has_prev=1, inb_prev_departed=int(departed), inb_prev_arrived=int(arrived),
                   inb_prev_sched_dep_before_cutoff=int(sched_before),
                   inb_prev_dep_delay=prev["dep_delay_minutes"] if departed else np.nan,
                   inb_prev_arr_delay=prev["arr_delay_minutes"] if arrived else np.nan,
                   inb_prev_overdue_min=overdue, inb_known_delay_min=known, inb_sched_turn_min=turn,
                   inb_projected_slack_min=turn - max(known, 0))

    def mean_delay(d, col, tcol):
        d = d[(d[tcol] >= c - cf.WINDOW_MIN) & (d[tcol] < c)]
        return (np.clip(d[col], *cf.CLIP).mean() if len(d) else np.nan), len(d)

    o_dep = dep_seen(by_origin_dep[r["origin"]])
    out["origin_recent_dep_delay_avg"], out["origin_recent_dep_count"] = mean_delay(o_dep, "dep_delay_minutes",
                                                                                    "actual_dep_utc_m")
    out["origin_recent_arr_delay_avg"], _ = mean_delay(arr_seen(by_dest_arr.get(r["origin"], ev.iloc[:0])),
                                                       "arr_delay_minutes", "actual_arr_utc_m")
    out["dest_recent_arr_delay_avg"], _ = mean_delay(arr_seen(by_dest_arr[r["dest"]]), "arr_delay_minutes",
                                                     "actual_arr_utc_m")
    out["dest_recent_dep_delay_avg"], _ = mean_delay(dep_seen(by_origin_dep.get(r["dest"], ev.iloc[:0])),
                                                     "dep_delay_minutes", "actual_dep_utc_m")
    out["carrier_recent_dep_delay_avg"], _ = mean_delay(dep_seen(by_carrier[r["carrier_code"]]),
                                                        "dep_delay_minutes", "actual_dep_utc_m")
    out["nat_recent_dep_delay_avg"], _ = mean_delay(ev[ev["actual_dep_utc_m"] < c], "dep_delay_minutes",
                                                    "actual_dep_utc_m")
    # pending: scheduled to leave the origin in [c-2h, c) and NOT seen departing before c
    sched = by_origin_dep[r["origin"]]
    sched = sched[(sched["sched_dep_utc_m"] >= c - cf.WINDOW_MIN) & (sched["sched_dep_utc_m"] < c)]
    pending = sched[~(sched["actual_dep_utc_m"] < c)]
    out["origin_pending_count"] = len(pending)
    out["origin_pending_share"] = len(pending) / len(sched) if len(sched) else np.nan
    return out


class _Groups:
    """Lazy per-key views of the events (index arrays, no full copies)."""
    def __init__(self, ev, col):
        self.ev = ev
        self.idx = {k: v for k, v in ev.groupby(col, observed=True).indices.items()}

    def __getitem__(self, k):
        return self.ev.iloc[self.idx[k]]

    def get(self, k, default):
        return self[k] if k in self.idx else default


def check_recompute(engine, failures, report):
    months, first = cf.model_window(engine)
    ev = cf.load_events(engine, first)
    ev["row"] = np.arange(len(ev))
    targets = np.flatnonzero((ev["in_window"] & ~ev["is_cancelled"] & ~ev["is_diverted"]).to_numpy())
    sample = np.sort(np.random.RandomState(SEED).choice(targets, size=min(SAMPLE, len(targets)), replace=False))
    keys = ev["flight_key"].to_numpy()[sample].tolist()
    feats = pd.read_sql(sqlalchemy.text("SELECT * FROM warehouse.flight_cutoff_features WHERE flight_key = ANY(:k)")
                        .bindparams(k=keys), engine).set_index("flight_key")
    ev_tail = ev[ev["tail_number"].notna()]
    by_tail = _Groups(ev_tail, "tail_number")
    by_origin_dep = _Groups(ev, "origin")
    by_dest_arr = _Groups(ev, "dest")
    by_carrier = _Groups(ev, "carrier_code")
    cols = cf.INBOUND_FEATURES + [c for c in cf.CONGESTION_FEATURES]
    mismatches = []
    for n_done, i in enumerate(sample):
        naive = naive_features(i, ev, by_tail, by_origin_dep, by_dest_arr, by_carrier)
        stored = feats.loc[ev.iloc[i]["flight_key"]]
        for col in cols:
            a, b = float(naive[col]), float(stored[col])
            if not ((np.isnan(a) and np.isnan(b)) or (abs(a - b) <= 1e-5 * max(1.0, abs(a)))):
                mismatches.append({"flight_key": ev.iloc[i]["flight_key"], "feature": col, "naive": a, "stored": b})
        if n_done % 100 == 0:
            print(f"   recomputed {n_done}/{len(sample)} flights", flush=True)
    report["independent_recompute"] = {"sample_flights": int(len(sample)), "features_compared": len(cols),
                                       "values_compared": int(len(sample) * len(cols)),
                                       "mismatches": len(mismatches), "examples": mismatches[:5]}
    print("3. independent recompute:", report["independent_recompute"], flush=True)
    if mismatches:
        failures.append(f"{len(mismatches)} naive-vs-stored mismatches")


def check_feature_names(failures, report):
    import train_delay_model as t
    bad = sorted(set(t.FEATURES) & FORBIDDEN_OWN_COLUMNS)
    report["feature_name_rule"] = {"n_features": len(t.FEATURES), "forbidden_used": bad}
    print("4. feature-name rule:", report["feature_name_rule"])
    if bad:
        failures.append(f"forbidden own post-departure columns used as features: {bad}")


def check_timezones(engine, failures, report):
    row = pd.read_sql("""
        SELECT count(*) AS flights,
               avg((abs(extract(epoch FROM (sched_arr_utc_local_clock - sched_dep_utc)) / 60 - crs_elapsed_minutes) <= 5)::int)
                   AS share_block_time_consistent
        FROM intermediate.int_flights_enriched WHERE crs_elapsed_minutes IS NOT NULL""", engine).iloc[0]
    report["timezone_sanity"] = {"flights": int(row["flights"]),
                                 "share_utc_block_time_within_5min": round(float(row["share_block_time_consistent"]), 5)}
    print("5. time-zone sanity:", report["timezone_sanity"])
    if row["share_block_time_consistent"] < 0.99:
        failures.append("UTC block times disagree with BTS scheduled elapsed time (time-zone problem)")


def main():
    engine = get_engine()
    failures, report = [], {}
    check_timestamps(engine, failures, report)
    check_keys(engine, failures, report)
    check_timezones(engine, failures, report)
    check_feature_names(failures, report)
    check_recompute(engine, failures, report)
    report["passed"] = not failures
    report["failures"] = failures
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "leakage_check.json").write_text(json.dumps(report, indent=2, default=str))
    if failures:
        print("LEAKAGE CHECK FAILED:", failures)
        sys.exit(1)
    print("LEAKAGE CHECK PASSED")


if __name__ == "__main__":
    main()
