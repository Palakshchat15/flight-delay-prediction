"""
Features known 2 HOURS BEFORE SCHEDULED DEPARTURE -> warehouse.flight_cutoff_features.

Prediction moment for flight i: cutoff_i = sched_dep_utc_i - 2 h (UTC, time-zone aware;
see int_flights_enriched). A feature may use the published schedule (known in advance)
and OTHER flights' actual events that happened strictly before cutoff_i. It never uses
flight i's own departure delay, taxi, wheels-off, actual times or delay-cause columns.

Event times (UTC minutes): a flight's departure delay becomes known at its actual
departure (actual_dep = sched_dep + dep_delay); its arrival delay at its actual arrival
(actual_arr = sched_arr + arr_delay). "Departed by the cutoff" means actual_dep < cutoff;
"arrived by the cutoff" means actual_arr < cutoff. Cancelled flights never depart and
diverted flights never arrive at their destination.

1. Inbound aircraft (same Tail_Number). "Previous leg" = the aircraft's flight with the
   latest scheduled departure before this one's, within the previous 24 h (cancelled legs
   included: they are part of the published rotation).
     inb_has_prev                     previous leg exists (schedule)
     inb_sched_turn_min               sched_dep - prev sched_arr (schedule)
     inb_prev_sched_dep_before_cutoff prev leg was scheduled to leave before the cutoff (schedule)
     inb_prev_departed                prev actual_dep < cutoff
     inb_prev_dep_delay               prev dep delay, ONLY if prev actual_dep < cutoff, else NULL
     inb_prev_arrived                 prev actual_arr < cutoff
     inb_prev_arr_delay               prev arr delay, ONLY if prev actual_arr < cutoff, else NULL
     inb_prev_overdue_min             prev scheduled to leave before the cutoff but has not left
                                      by it: cutoff - prev sched_dep (a lower bound on its delay
                                      that is observable at the cutoff); else 0
     inb_known_delay_min              best delay estimate for the inbound at the cutoff:
                                      arrival delay if arrived; else max(dep delay, cutoff -
                                      prev sched_arr) if departed; else the overdue minutes; else 0
     inb_projected_slack_min          inb_sched_turn_min - max(inb_known_delay_min, 0)
     ac_latest_dep_delay              dep delay of the aircraft's most recent leg that departed
                                      in the 24 h before the cutoff (actual_dep < cutoff)
     ac_latest_arr_delay              arr delay of its most recent leg that arrived in the
                                      24 h before the cutoff (actual_arr < cutoff)
2. Congestion in the 2 hours before the cutoff, window [cutoff - 2 h, cutoff):
     origin_recent_dep_count / _dep_delay_avg   departures from the origin with actual_dep in window
     origin_pending_count / _share              flights scheduled to leave the origin in the window
                                                that have not departed by the cutoff (count, share)
     origin_recent_arr_delay_avg                arrivals at the origin with actual_arr in window
     dest_recent_arr_delay_avg / _dep_delay_avg arrivals at / departures from the destination
     carrier_recent_dep_delay_avg               the carrier's departures network-wide in window
     nat_recent_dep_delay_avg                   all departures network-wide in window
   (delays averaged after clipping to [-30, 300] min; NULL when the window is empty)
3. Weather (Open-Meteo, UTC hours; hour T holds values at T / sums over the hour ending at T):
     wx_o_*   origin weather at the last full hour <= cutoff (known at the cutoff)
     wx_o_precip_3h  origin precipitation over the 3 hours ending at that hour
     wx_d_*   destination weather at the scheduled arrival hour: OBSERVED weather standing in
              for a forecast (a real forecast would be less accurate; README limitation)

For the leakage check every row also stores the time of the latest event each feature
group used (*_ts) and the keys of the legs used; leakage_check.py and the dbt singular
tests assert *_ts < cutoff_utc and recompute a sample independently.
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db_utils import (concat_columns, copy_text_column, get_engine, read_sql_chunks,  # noqa: E402
                      release_memory, shared_objects)

CUTOFF_MIN = 120          # prediction point: 2 h before scheduled departure
WINDOW_MIN = 120          # congestion window before the cutoff
LOOKBACK_MIN = 24 * 60    # inbound aircraft lookback
CLIP = (-30, 300)
MODEL_WINDOW_MONTHS = int(os.environ.get("MODEL_WINDOW_MONTHS", "4"))
EPOCH = np.datetime64("1970-01-01T00:00")
EVENT_TS = ("sched_dep_utc", "sched_arr_utc", "actual_dep_utc", "actual_arr_utc", "cutoff_utc")
WX_VARS = {"temperature_c": "temp", "precipitation_mm": "precip", "snowfall_cm": "snow",
           "wind_speed_kmh": "wind", "wind_gusts_kmh": "gust", "cloud_cover_low_pct": "cloud_low"}

INBOUND_FEATURES = [
    "inb_has_prev", "inb_sched_turn_min", "inb_prev_sched_dep_before_cutoff", "inb_prev_departed",
    "inb_prev_dep_delay", "inb_prev_arrived", "inb_prev_arr_delay", "inb_prev_overdue_min",
    "inb_known_delay_min", "inb_projected_slack_min", "ac_latest_dep_delay", "ac_latest_arr_delay",
]
CONGESTION_FEATURES = [
    "origin_recent_dep_count", "origin_recent_dep_delay_avg", "origin_pending_count", "origin_pending_share",
    "origin_recent_arr_delay_avg", "dest_recent_arr_delay_avg", "dest_recent_dep_delay_avg",
    "carrier_recent_dep_delay_avg", "nat_recent_dep_delay_avg",
]
WEATHER_ORIGIN_FEATURES = [f"wx_o_{v}" for v in WX_VARS.values()] + ["wx_o_precip_3h"]
WEATHER_DEST_FEATURES = [f"wx_d_{v}" for v in WX_VARS.values()]
CUTOFF_FEATURES = INBOUND_FEATURES + CONGESTION_FEATURES + WEATHER_ORIGIN_FEATURES + WEATHER_DEST_FEATURES
# event-time columns that must be < cutoff (the destination weather hour is the forecast stand-in)
STRICT_TS = ["inb_prev_dep_ts", "inb_prev_arr_ts", "ac_latest_dep_ts", "ac_latest_arr_ts",
             "origin_dep_win_max_ts", "origin_arr_win_max_ts", "dest_arr_win_max_ts", "dest_dep_win_max_ts",
             "carrier_win_max_ts", "nat_win_max_ts"]
KEY_COLS = ["prev_flight_key", "ac_latest_dep_key", "ac_latest_arr_key"]
INT_FEATURES = ["inb_has_prev", "inb_prev_sched_dep_before_cutoff", "inb_prev_departed", "inb_prev_arrived",
                "origin_recent_dep_count", "origin_pending_count"]

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:6.0f}s] {msg}", flush=True)


def to_min(s):
    """timestamp Series -> float minutes since epoch (NaN for NaT)."""
    v = s.to_numpy(dtype="datetime64[m]")
    out = (v - EPOCH).astype("timedelta64[m]").astype("float64")
    out[np.isnat(v)] = np.nan
    return out


def from_min(x):
    """Event times stay as float minutes since epoch (NaN = none) until they are written."""
    return np.asarray(x, dtype="float64")


def min_to_ts(x):
    x = np.asarray(x, dtype="float64")
    out = np.full(len(x), np.datetime64("NaT"), dtype="datetime64[m]")
    ok = ~np.isnan(x)
    out[ok] = EPOCH + x[ok].astype("int64").astype("timedelta64[m]")
    return pd.to_datetime(out)


def model_window(engine):
    months = pd.read_sql("SELECT DISTINCT flight_year, flight_month FROM intermediate.int_flights_enriched "
                         "ORDER BY 1, 2", engine)
    months = months.tail(MODEL_WINDOW_MONTHS)
    first = pd.Timestamp(int(months.iloc[0, 0]), int(months.iloc[0, 1]), 1)
    return months, first


def _reduce_event_chunk(chunk, canon):
    """Timestamps -> the float minutes that are the only form used; delays float64 with NULL =
    NaN (as a one-shot read gives); one shared object per repeated code/date."""
    for c in EVENT_TS:
        chunk[c + "_m"] = to_min(pd.to_datetime(chunk.pop(c)))
    for c in ("dep_delay_minutes", "arr_delay_minutes"):
        chunk[c] = chunk[c].astype("float64")
    for c in ("flight_date", "carrier_code", "tail_number", "origin", "dest"):
        chunk[c] = shared_objects(chunk[c], canon.setdefault(c, {}))
    return chunk


def load_events(engine, first_month_start):
    # Two days of earlier flights so the first cutoffs of the window see their full lookback.
    rows = f"""
        FROM intermediate.int_flights_enriched
        WHERE flight_date >= DATE '{first_month_start.date()}' - 2
        ORDER BY flight_date, flight_key"""
    # Memory (4 GB Docker VM): a one-shot pd.read_sql held all ~2.3M rows as Python objects
    # (~2.7 GB) and was OOM-killed. Same rows, order and values, read as: flight_key (unique,
    # the only per-row string) first, via COPY in the same ORDER BY on a unique key; then the
    # other columns streamed in chunks (server-side cursor), each chunk reduced as it arrives
    # and the chunks joined column by column (db_utils.concat_columns).
    keys = copy_text_column(engine, f"SELECT flight_key {rows}")
    canon = {}
    ev = pd.DataFrame({"flight_key": keys, **concat_columns(
        _reduce_event_chunk(chunk, canon) for chunk in read_sql_chunks(engine, f"""
            SELECT flight_date, flight_year, flight_month, carrier_code, tail_number, origin, dest,
                   is_cancelled, is_diverted, dep_delay_minutes, arr_delay_minutes,
                   sched_dep_utc, sched_arr_utc, actual_dep_utc, actual_arr_utc, cutoff_utc {rows}"""))},
                      copy=False)
    assert len(keys) == len(ev), "flight_key read must match the event rows"
    del keys
    release_memory()
    for c in ("carrier_code", "tail_number", "origin", "dest"):
        ev[c] = ev[c].astype("category")
    ev["in_window"] = ev["flight_date"] >= first_month_start.date()
    return ev


def load_weather(engine, first_month_start):
    sql = (f"SELECT airport_code, ts_utc, {', '.join(WX_VARS)} FROM raw.weather_hourly "
           f"WHERE ts_utc >= TIMESTAMP '{first_month_start.date()}' - interval '3 days' ORDER BY airport_code, ts_utc")
    canon = {}
    wx = pd.DataFrame(concat_columns(chunk.assign(airport_code=shared_objects(chunk["airport_code"], canon))
                                     for chunk in read_sql_chunks(engine, sql)), copy=False)
    release_memory()
    return wx


class TargetColumns(dict):
    """build()'s output columns, kept for the target rows only and in their final dtype (int32 /
    float32 features; keys and event times unchanged) as soon as each one is computed. Same values
    as converting at the end, without holding ~1 GB of full-length float64 columns at the peak."""

    def __init__(self, targets):
        super().__init__()
        self.targets = targets

    def __setitem__(self, name, values):
        v = np.asarray(values)[self.targets]
        super().__setitem__(name, v.astype("int32") if name in INT_FEATURES else
                            v.astype("float32") if name in CUTOFF_FEATURES else v)


def window_stats(group_codes, times, values, q_codes, q_end, width=WINDOW_MIN):
    """For each query (code, end): count, mean of `values` and max event time over events of the
    same code with time in [end - width, end). Events with NaN time are ignored."""
    ok = ~np.isnan(times)
    g, t, v = group_codes[ok].astype("int64"), times[ok], values[ok]
    order = np.lexsort((t, g))
    g, t, v = g[order], t[order], v[order]
    key = g * 1e9 + t
    csum = np.concatenate([[0.0], np.cumsum(v)])
    hi = np.searchsorted(key, q_codes * 1e9 + q_end, side="left")
    lo = np.searchsorted(key, q_codes * 1e9 + q_end - width, side="left")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
    last_t = np.where(n > 0, t[np.maximum(hi - 1, 0)], np.nan)
    return n, mean, last_t


def latest_before(group_codes, times, idx, q_codes, q_time, lookback=LOOKBACK_MIN):
    """Index (into the original arrays) of the event of the same code with the latest time
    in [q_time - lookback, q_time); -1 if none."""
    ok = ~np.isnan(times)
    g, t, ix = group_codes[ok].astype("int64"), times[ok], idx[ok]
    order = np.lexsort((ix, t, g))
    g, t, ix = g[order], t[order], ix[order]
    key = g * 1e9 + t
    pos = np.searchsorted(key, q_codes * 1e9 + q_time, side="left") - 1
    valid = pos >= 0
    posc = np.maximum(pos, 0)
    valid &= (g[posc] == q_codes) & (t[posc] >= q_time - lookback)
    return np.where(valid, ix[posc], -1)


def pending_counts(group_codes, sched, actual, q_codes, q_time, width=WINDOW_MIN):
    """Flights of the same code scheduled in [c - width, c) that have NOT departed by c
    (actual >= c, or never departed). Each flight is 'pending' for c in (sched, end] with
    end = max(sched, min(sched + width, actual)); count = #(sched < c) - #(end < c)."""
    act = np.where(np.isnan(actual), np.inf, actual)
    end = np.maximum(sched, np.minimum(sched + width, act))
    g = group_codes.astype("int64")
    ks = np.sort(g * 1e9 + sched)
    ke = np.sort(g * 1e9 + end)
    q = q_codes * 1e9 + q_time
    return np.searchsorted(ks, q, side="left") - np.searchsorted(ke, q, side="left")


def build(ev, wx, targets):
    n_ev = len(ev)
    idx = np.arange(n_ev)
    cut = ev["cutoff_utc_m"].to_numpy()
    s_dep, s_arr = ev["sched_dep_utc_m"].to_numpy(), ev["sched_arr_utc_m"].to_numpy()
    a_dep, a_arr = ev["actual_dep_utc_m"].to_numpy(), ev["actual_arr_utc_m"].to_numpy()
    dep_delay = ev["dep_delay_minutes"].to_numpy(dtype="float64")
    arr_delay = ev["arr_delay_minutes"].to_numpy(dtype="float64")
    # Post-event values exist only where the event happened.
    assert np.all(np.isnan(a_dep) == np.isnan(dep_delay)) and np.all(np.isnan(a_arr) == np.isnan(arr_delay))
    # A flight cannot have departed before its own cutoff (2 h early); if it had, its own
    # event could leak into its "aircraft latest leg" features.
    assert not np.any(a_dep < cut), "a flight departed more than 2 h before its schedule"
    assert not np.any(a_arr < cut), "a flight 'arrived' before its own cutoff (arrival-time bug)"
    out = TargetColumns(targets)
    out["flight_key"] = ev["flight_key"].to_numpy()
    out["cutoff_utc"] = cut

    # ---- 1. inbound aircraft
    tail = ev["tail_number"].astype(object).fillna("").to_numpy()
    has_tail = tail != ""
    tail_code = pd.factorize(tail)[0].astype("int64")
    tail_code[~has_tail] = -1
    tq = np.where(has_tail, tail_code, -2)  # flights without a tail match nothing
    # previous leg = the tail's leg with the latest STRICTLY earlier scheduled departure within
    # 24 h (legs at the same minute are a data artefact, not a rotation); ties -> later row
    prev = latest_before(tail_code, np.where(has_tail, s_dep, np.nan), idx, tq, s_dep)
    p_ok = prev >= 0
    prev = np.where(p_ok, prev, -1)
    pi = np.maximum(prev, 0)

    def p(a):
        return np.where(p_ok, a[pi], np.nan)

    p_sdep, p_sarr, p_adep, p_aarr = p(s_dep), p(s_arr), p(a_dep), p(a_arr)
    p_depd, p_arrd = p(dep_delay), p(arr_delay)
    departed = p_ok & (p_adep < cut)          # NaN comparisons are False: never departed
    arrived = p_ok & (p_aarr < cut)
    sched_before = p_ok & (p_sdep < cut)
    overdue = np.where(sched_before & ~departed, cut - p_sdep, 0.0)
    late_vs_sched_arr = np.clip(cut - p_sarr, 0, None)
    known = np.where(arrived, p_arrd,
                     np.where(departed, np.fmax(p_depd, late_vs_sched_arr),
                              np.where(sched_before, overdue, 0.0)))
    out["inb_has_prev"] = p_ok.astype(int)
    out["inb_sched_turn_min"] = np.where(p_ok, s_dep - p_sarr, np.nan)
    out["inb_prev_sched_dep_before_cutoff"] = sched_before.astype(int)
    out["inb_prev_departed"] = departed.astype(int)
    out["inb_prev_dep_delay"] = np.where(departed, p_depd, np.nan)
    out["inb_prev_arrived"] = arrived.astype(int)
    out["inb_prev_arr_delay"] = np.where(arrived, p_arrd, np.nan)
    out["inb_prev_overdue_min"] = overdue
    out["inb_known_delay_min"] = np.where(p_ok, known, 0.0)
    out["inb_projected_slack_min"] = np.where(p_ok, (s_dep - p_sarr) - np.maximum(known, 0), np.nan)
    out["prev_flight_key"] = np.where(p_ok, ev["flight_key"].to_numpy()[pi], None)
    out["inb_prev_dep_ts"] = from_min(np.where(departed, p_adep, np.nan))
    out["inb_prev_arr_ts"] = from_min(np.where(arrived, p_aarr, np.nan))

    for kind, times, delay in (("dep", a_dep, dep_delay), ("arr", a_arr, arr_delay)):
        t = np.where(has_tail, times, np.nan)
        j = latest_before(tail_code, t, idx, tq, cut)
        ok = j >= 0
        jj = np.maximum(j, 0)
        out[f"ac_latest_{kind}_delay"] = np.where(ok, delay[jj], np.nan)
        out[f"ac_latest_{kind}_ts"] = from_min(np.where(ok, times[jj], np.nan))
        out[f"ac_latest_{kind}_key"] = np.where(ok, ev["flight_key"].to_numpy()[jj], None)
    # free this section's full-length intermediates (~0.4 GB) before the next one
    del p, tail, has_tail, tail_code, tq, prev, p_ok, pi, p_sdep, p_sarr, p_adep, p_aarr, p_depd, p_arrd, \
        departed, arrived, sched_before, overdue, late_vs_sched_arr, known, t, j, ok, jj
    log("inbound aircraft features done")

    # ---- 2. congestion in [cutoff - 2 h, cutoff)
    airports = pd.Index(sorted(set(ev["origin"]) | set(ev["dest"])))
    o_code = airports.get_indexer(ev["origin"]).astype("int64")
    d_code = airports.get_indexer(ev["dest"]).astype("int64")
    c_code = pd.factorize(ev["carrier_code"])[0].astype("int64")
    zero = np.zeros(n_ev, dtype="int64")
    dd, ad = np.clip(dep_delay, *CLIP), np.clip(arr_delay, *CLIP)
    n, m, t = window_stats(o_code, a_dep, dd, o_code, cut)
    out["origin_recent_dep_count"], out["origin_recent_dep_delay_avg"], out["origin_dep_win_max_ts"] = n, m, from_min(t)
    _, m, t = window_stats(d_code, a_arr, ad, o_code, cut)          # arrivals INTO the origin
    out["origin_recent_arr_delay_avg"], out["origin_arr_win_max_ts"] = m, from_min(t)
    _, m, t = window_stats(d_code, a_arr, ad, d_code, cut)          # arrivals into the destination
    out["dest_recent_arr_delay_avg"], out["dest_arr_win_max_ts"] = m, from_min(t)
    _, m, t = window_stats(o_code, a_dep, dd, d_code, cut)          # departures from the destination
    out["dest_recent_dep_delay_avg"], out["dest_dep_win_max_ts"] = m, from_min(t)
    _, m, t = window_stats(c_code, a_dep, dd, c_code, cut)
    out["carrier_recent_dep_delay_avg"], out["carrier_win_max_ts"] = m, from_min(t)
    _, m, t = window_stats(zero, a_dep, dd, zero, cut)
    out["nat_recent_dep_delay_avg"], out["nat_win_max_ts"] = m, from_min(t)
    pend = pending_counts(o_code, s_dep, a_dep, o_code, cut)
    sched_n = pending_counts(o_code, s_dep, np.full(n_ev, np.inf), o_code, cut)  # all scheduled in window
    out["origin_pending_count"] = pend
    out["origin_pending_share"] = np.where(sched_n > 0, pend / np.maximum(sched_n, 1), np.nan)
    del c_code, zero, dd, ad, n, m, t, pend, sched_n
    log("congestion features done")

    # ---- 3. weather
    wx = wx.rename(columns=WX_VARS)
    wx["hour_m"] = to_min(wx["ts_utc"])
    wx["a"] = airports.get_indexer(wx["airport_code"])
    wx = wx[wx["a"] >= 0]
    wkey = pd.Series(np.arange(len(wx)), index=wx["a"].to_numpy() * 1e9 + wx["hour_m"].to_numpy())
    wkey = wkey[~wkey.index.duplicated()]
    wvals = {v: wx[v].to_numpy(dtype="float64") for v in WX_VARS.values()}
    del wx  # the renamed / filtered copies; wkey and wvals hold what the lookups need

    def wx_lookup(codes, hour_m):
        pos = wkey.reindex(codes * 1e9 + hour_m).to_numpy()
        ok = ~np.isnan(pos)
        posi = np.where(ok, pos, 0).astype("int64")
        return ok, posi

    o_hour = np.floor(cut / 60) * 60                       # last full hour <= cutoff
    ok, pos = wx_lookup(o_code, o_hour)
    for v, arr in wvals.items():
        out[f"wx_o_{v}"] = np.where(ok, arr[pos], np.nan)
    out["wx_o_ts"] = from_min(np.where(ok, o_hour, np.nan))
    precip3 = np.zeros(n_ev)
    n3 = np.zeros(n_ev)
    for k in range(3):
        okk, posk = wx_lookup(o_code, o_hour - 60 * k)
        val = np.where(okk, wvals["precip"][posk], np.nan)
        precip3 += np.nan_to_num(val)
        n3 += ~np.isnan(val)
    out["wx_o_precip_3h"] = np.where(n3 == 3, precip3, np.nan)
    d_hour = np.floor(s_arr / 60) * 60                     # scheduled arrival hour (forecast stand-in)
    ok, pos = wx_lookup(d_code, d_hour)
    for v, arr in wvals.items():
        out[f"wx_d_{v}"] = np.where(ok, arr[pos], np.nan)
    out["wx_d_ts"] = from_min(np.where(ok, d_hour, np.nan))
    log("weather features done")
    return out


def write_table(columns, engine):
    """columns: {name: array} in table order. Written in 100k-row pieces, each built as a small
    DataFrame: one 2M-row CSV buffer would need ~1.5 GB, a full 2M-row frame another ~0.7 GB."""
    names = list(columns)
    n_rows = len(columns[names[0]])
    ts_cols = [c for c in names if c.endswith("_ts") or c == "cutoff_utc"]
    int_cols = INT_FEATURES
    cols = []
    for c in names:
        typ = "TIMESTAMP" if c in ts_cols else "TEXT" if c in ["flight_key"] + KEY_COLS else \
            "INTEGER" if c in int_cols else "DOUBLE PRECISION"
        cols.append(f"{c} {typ}")
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS warehouse.flight_cutoff_features")
        conn.exec_driver_sql(f"CREATE TABLE warehouse.flight_cutoff_features ({', '.join(cols)}, "
                             "PRIMARY KEY (flight_key))")
        cur = conn.connection.cursor()
        for i in range(0, n_rows, 100_000):
            part = pd.DataFrame({c: v[i:i + 100_000] for c, v in columns.items()})
            for c in ts_cols:
                part[c] = min_to_ts(part[c].to_numpy())
            buf = io.StringIO()
            part.to_csv(buf, index=False, header=False, date_format="%Y-%m-%d %H:%M:%S")
            buf.seek(0)
            cur.copy_expert("COPY warehouse.flight_cutoff_features FROM STDIN WITH (FORMAT csv, NULL '')", buf)
        conn.exec_driver_sql("ANALYZE warehouse.flight_cutoff_features")


def main():
    engine = get_engine()
    months, first = model_window(engine)
    log(f"Model window months: {[f'{y}-{m:02d}' for y, m in months.to_numpy()]} (events from {first.date()} - 2 days)")
    ev = load_events(engine, first)
    missing_utc = np.isnan(ev["sched_dep_utc_m"]).sum() + np.isnan(ev["sched_arr_utc_m"]).sum()
    assert missing_utc == 0, f"{missing_utc} flights without a UTC schedule (missing airport time zone)"
    wx = load_weather(engine, first)
    log(f"Loaded {len(ev):,} flights (events) and {len(wx):,} hourly weather rows")
    targets = (ev["in_window"] & ~ev["is_cancelled"] & ~ev["is_diverted"]).to_numpy()
    # Target rows only, features as int32 / float32 (TargetColumns). Written from these arrays
    # directly; a DataFrame of them would be one more full copy.
    feats = build(ev, wx, targets)
    del ev, wx
    for c in ["inb_has_prev", "inb_prev_departed", "inb_prev_arrived", "inb_prev_sched_dep_before_cutoff"]:
        log(f"  {c}: {pd.Series(feats[c]).mean():.3f}")
    for c in ["inb_prev_dep_delay", "inb_prev_arr_delay", "ac_latest_dep_delay", "origin_recent_dep_delay_avg",
              "wx_o_precip", "wx_d_wind"]:
        f = pd.Series(feats[c])
        log(f"  {c}: non-null {f.notna().mean():.3f}, mean {f.mean():.2f}")
    write_table(feats, engine)
    log(f"Wrote {len(feats['flight_key']):,} rows x {len(CUTOFF_FEATURES)} features to warehouse.flight_cutoff_features")


if __name__ == "__main__":
    main()
