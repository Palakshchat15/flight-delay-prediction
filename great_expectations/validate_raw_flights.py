"""
Great Expectations validation of raw.flights (BTS On-Time Performance).

Same lightweight pattern as the sibling projects: read the table into pandas
and run an in-memory expectation suite (GE 0.18 PandasDataset) instead of a
full Data Context / checkpoint project. Exits non-zero on any hard-gate
failure so the Airflow task fails and blocks dbt/model training.

MEMORY (runs in a 4 GB Docker VM): all 2.2M rows x 13 columns in one pandas
frame need ~2.6 GB and were OOM-killed there, so the table is validated in
bounded pieces with exactly the same expectations and pass/fail result:
  - row-level expectations (not null, regex, in set, between without `mostly`,
    compound-key uniqueness) run on blocks of CHUNK_DAYS flight dates. A block
    holds every row of its dates, and flight_date is part of the duplicate key,
    so no duplicate can span two blocks; each expectation passes only if it
    passes on every block, and the unexpected counts are summed;
  - whole-table expectations (the cancellation-rate mean and the `mostly`
    arrival-delay check) run once on all rows, loading only the numeric
    columns they need;
  - rows per monthly file are counted per block and summed.

BUSINESS RULES (what is NOT an error):
  - Cancelled flights (cancelled = 1) never depart or arrive, so dep_time,
    arr_time, arr_delay and arr_del15 are legitimately NULL for them.
  - Diverted flights (diverted = 1) land somewhere else, so arr_delay /
    arr_del15 are legitimately NULL for them too.
  - tail_number is NULL for some cancelled flights (no aircraft assigned).
  These rows are kept in the warehouse (they drive the cancellation-rate KPI)
  and excluded only from the delay model, whose label needs an arrival.

HARD GATES (structural integrity; any failure blocks the pipeline):
  - required keys not null; airport codes are 3-character IATA codes;
    scheduled times are valid hhmm (BTS's "2400" = end-of-day midnight is
    allowed; dbt staging maps it to 2359); cancelled/diverted flags are 0/1;
    distance within a plausible domestic range; no duplicate flights
    (date + carrier + flight number + origin + scheduled departure);
  - every COMPLETED flight (not cancelled, not diverted) has an arrival delay
    and arr_del15 agrees with arr_delay >= 15;
  - each monthly file loaded between 400k and 800k rows (BTS months are ~550k).

WARN-ONLY (measured and printed every run, not blocking):
  - overall cancellation rate between 0.5% and 8% (a winter storm month can
    legitimately spike, so this is a signal to look at, not a data error);
  - arrival delay within [-120, 2000] minutes for 99.99% of completed flights.
"""
import os
import sys
from types import SimpleNamespace

import great_expectations as ge
import pandas as pd
import sqlalchemy


def get_engine():
    user = os.environ.get("POSTGRES_USER", "flight_user")
    password = os.environ.get("POSTGRES_PASSWORD", "flight_pass")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5436")
    db = os.environ.get("POSTGRES_DB", "flight_delay")
    return sqlalchemy.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")


# NUMERIC columns are cast so pandas gets floats/ints rather than Decimal objects.
COLUMNS = ["flight_date", "reporting_airline", "flight_number", "origin", "dest",
           "crs_dep_time", "crs_arr_time", "arr_delay::float8 AS arr_delay",
           "arr_del15::float8 AS arr_del15", "cancelled::int AS cancelled",
           "diverted::int AS diverted", "distance::float8 AS distance", "source_file"]
# Only what the whole-table expectations and the printed rates need (cheap numeric columns).
NUMERIC_COLUMNS = ["arr_delay::float8 AS arr_delay", "arr_del15::float8 AS arr_del15",
                   "cancelled::int AS cancelled", "diverted::int AS diverted"]
# BTS writes end-of-day midnight as "2400" (1 scheduled departure in Jan-Apr 2024); it is valid.
HHMM = r"^(([01][0-9]|2[0-3])[0-5][0-9]|2400)$"
CHUNK_DAYS = 10  # ~185k rows per block


def read_chunked(engine, sql, chunksize=200_000):
    """pd.read_sql through a server-side cursor, concatenated (never all rows as Python tuples)."""
    with engine.connect().execution_options(stream_results=True) as conn:
        return pd.concat(pd.read_sql(sqlalchemy.text(sql), conn, chunksize=chunksize), ignore_index=True)


def completed_flights(df):
    completed = df[(df["cancelled"] == 0) & (df["diverted"] == 0)].copy()
    completed["late_by_delay"] = (completed["arr_delay"] >= 15).astype(float)
    completed["del15_matches"] = completed["late_by_delay"] == completed["arr_del15"]
    return completed


def row_level_expectations(g_all, g_done):
    """The row-level hard gates, in report order; each passes on the table iff it passes on every
    date block (map expectations without `mostly`; the duplicate key contains flight_date)."""
    out = []
    for col in ["flight_date", "reporting_airline", "flight_number", "origin", "dest",
                "crs_dep_time", "crs_arr_time", "cancelled", "diverted", "distance"]:
        out.append((f"{col}_not_null", g_all.expect_column_values_to_not_be_null(col)))
    for col in ["origin", "dest"]:
        out.append((f"{col}_is_iata_code", g_all.expect_column_values_to_match_regex(col, r"^[A-Z0-9]{3}$")))
    for col in ["crs_dep_time", "crs_arr_time"]:
        out.append((f"{col}_is_valid_hhmm", g_all.expect_column_values_to_match_regex(col, HHMM)))
    for col in ["cancelled", "diverted"]:
        out.append((f"{col}_is_flag", g_all.expect_column_values_to_be_in_set(col, [0, 1])))
    out.append(("distance_plausible", g_all.expect_column_values_to_be_between("distance", 10, 6000)))
    out.append(("no_duplicate_flights", g_all.expect_compound_columns_to_be_unique(
        ["flight_date", "reporting_airline", "flight_number", "origin", "crs_dep_time"])))
    out.append(("completed_flights_have_arr_delay", g_done.expect_column_values_to_not_be_null("arr_delay")))
    out.append(("completed_flights_have_arr_del15", g_done.expect_column_values_to_not_be_null("arr_del15")))
    out.append(("arr_del15_consistent_with_arr_delay",
                g_done.expect_column_values_to_be_in_set("del15_matches", [True])))
    return out


def combine(results):
    """One result per expectation from its per-block results: passes iff every block passes."""
    first = results[0]
    return SimpleNamespace(
        success=all(r.success for r in results),
        expectation_config=first.expectation_config,
        result={"unexpected_count": sum(r.result.get("unexpected_count") or 0 for r in results),
                "partial_unexpected_list": [v for r in results
                                            for v in r.result.get("partial_unexpected_list", [])][:20]})


def date_blocks(engine):
    """WHERE clauses covering every row once: CHUNK_DAYS distinct flight dates each (+ NULL dates)."""
    dates = pd.read_sql("SELECT DISTINCT flight_date FROM raw.flights ORDER BY 1", engine)["flight_date"]
    has_null = dates.isna().any()
    dates = dates.dropna().tolist()
    blocks = [f"flight_date BETWEEN DATE '{dates[i]}' AND DATE '{dates[min(i + CHUNK_DAYS, len(dates)) - 1]}'"
              for i in range(0, len(dates), CHUNK_DAYS)]
    return blocks + (["flight_date IS NULL"] if has_null else [])


def main():
    engine = get_engine()
    num = read_chunked(engine, f"SELECT {', '.join(NUMERIC_COLUMNS)} FROM raw.flights")
    print(f"Loaded {len(num):,} rows from raw.flights for validation")
    if num.empty:
        print("FAIL: raw.flights is empty")
        sys.exit(1)

    num_done = completed_flights(num)
    print(f"Cancelled: {(num['cancelled'] == 1).sum():,} | Diverted: {(num['diverted'] == 1).sum():,} "
          f"| Completed: {len(num_done):,}")

    per_block, file_counts, n_rows = [], [], 0
    for where in date_blocks(engine):
        df = pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM raw.flights WHERE {where}", engine)
        n_rows += len(df)
        per_block.append(row_level_expectations(ge.from_pandas(df), ge.from_pandas(completed_flights(df))))
        file_counts.append(df.groupby("source_file").size())
        del df
    assert n_rows == len(num), "date blocks must cover every row exactly once"
    rows_per_file = pd.concat(file_counts).groupby(level=0).sum().rename("n").reset_index()
    g_files = ge.from_pandas(rows_per_file)

    hard = [(name, combine([block[i][1] for block in per_block])) for i, (name, _) in enumerate(per_block[0])]
    hard.append(("rows_per_monthly_file", g_files.expect_column_values_to_be_between("n", 400_000, 800_000)))

    warn = [
        ("cancellation_rate_plausible", ge.from_pandas(num).expect_column_mean_to_be_between("cancelled", 0.005, 0.08)),
        ("arr_delay_plausible", ge.from_pandas(num_done).expect_column_values_to_be_between(
            "arr_delay", -120, 2000, mostly=0.9999)),
    ]

    print("\nRows per monthly file:")
    print(rows_per_file.to_string(index=False))
    print(f"\nCancellation rate: {num['cancelled'].mean():.2%} | "
          f"diversion rate: {num['diverted'].mean():.2%} | "
          f"delayed (ArrDel15) rate among completed: {num_done['arr_del15'].mean():.2%}")

    failed = [n for n, r in hard if not r.success]
    print(f"\nRan {len(hard) + len(warn)} expectations ({len(hard)} hard-gate, {len(warn)} warn-only)")
    for name, r in hard:
        print(f"  [{'PASS' if r.success else 'FAIL'}] {name} ({r.expectation_config.expectation_type})")
        if not r.success:
            print(f"        unexpected_count={r.result.get('unexpected_count')} "
                  f"sample={r.result.get('partial_unexpected_list', [])[:5]}")
    for name, r in warn:
        detail = r.result.get("observed_value", r.result.get("unexpected_count"))
        print(f"  [{'PASS' if r.success else 'WARN (non-blocking)'}] {name}: {detail}")

    if failed:
        print(f"\nGreat Expectations validation FAILED: {failed}")
        sys.exit(1)
    print("\nGreat Expectations validation PASSED. Cancelled/diverted flights were allowed "
          "to have NULL arrival fields (business events, not errors).")


if __name__ == "__main__":
    main()
