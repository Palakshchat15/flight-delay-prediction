"""
Chunked ingestion of BTS "Reporting Carrier On-Time Performance (1987-present)"
monthly files into raw.flights, plus OurAirports reference data into raw.airports.

Source (official, public, no account needed):
  https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip
  https://davidmegginson.github.io/ourairports-data/airports.csv

Each monthly zip (~27 MB) holds one ~250 MB CSV with 110 columns. If a zip is
missing from data/raw/ it is downloaded; every zip is checked to be a real zip
containing a CSV before loading. Only the ~31 needed columns are read
(pandas usecols), in chunks, and streamed into Postgres with COPY.

The load is a full refresh (TRUNCATE + reload) so re-running the DAG is idempotent.
"""
import argparse
import io
import logging
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
BTS_URL = ("https://transtats.bts.gov/PREZIP/"
           "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# source column -> raw.flights column
COLUMN_MAP = {
    "FlightDate": "flight_date",
    "Year": "year",
    "Month": "month",
    "DayofMonth": "day_of_month",
    "DayOfWeek": "day_of_week",
    "Reporting_Airline": "reporting_airline",
    "Tail_Number": "tail_number",
    "Flight_Number_Reporting_Airline": "flight_number",
    "Origin": "origin",
    "OriginCityName": "origin_city_name",
    "OriginState": "origin_state",
    "Dest": "dest",
    "DestCityName": "dest_city_name",
    "DestState": "dest_state",
    "CRSDepTime": "crs_dep_time",
    "DepTime": "dep_time",
    "DepDelay": "dep_delay",
    "CRSArrTime": "crs_arr_time",
    "ArrTime": "arr_time",
    "ArrDelay": "arr_delay",
    "ArrDel15": "arr_del15",
    "Cancelled": "cancelled",
    "CancellationCode": "cancellation_code",
    "Diverted": "diverted",
    "CRSElapsedTime": "crs_elapsed_time",
    "Distance": "distance",
    "CarrierDelay": "carrier_delay",
    "WeatherDelay": "weather_delay",
    "NASDelay": "nas_delay",
    "SecurityDelay": "security_delay",
    "LateAircraftDelay": "late_aircraft_delay",
}
RAW_COLUMNS = list(COLUMN_MAP.values()) + ["source_file"]
# Read everything as text: hhmm times keep their leading zeros ("0856") and
# Postgres does the numeric casts on COPY.
DTYPES = {c: str for c in COLUMN_MAP}


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5436"),
        user=os.environ.get("POSTGRES_USER", "flight_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "flight_pass"),
        dbname=os.environ.get("POSTGRES_DB", "flight_delay"),
    )


def download(url, dest):
    logger.info("Downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (portfolio data pipeline)"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def ensure_month_zip(year, month):
    path = RAW_DIR / f"bts_ontime_{year}_{month}.zip"
    if not path.exists():
        download(BTS_URL.format(year=year, month=month), path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a valid zip (the server may have returned an HTML page)")
    with zipfile.ZipFile(path) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise ValueError(f"{path} should contain exactly one CSV, found {csvs}")
    return path, csvs[0]


def copy_frame(cur, df, table, columns):
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cur.copy_expert(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)


def load_flights(conn, year, months, chunksize):
    cur = conn.cursor()
    logger.info("Truncating raw.flights before full reload")
    cur.execute("TRUNCATE TABLE raw.flights")
    conn.commit()
    total = 0
    for month in months:
        path, csv_name = ensure_month_zip(year, month)
        month_rows = 0
        with zipfile.ZipFile(path) as z, z.open(csv_name) as f:
            reader = pd.read_csv(f, usecols=list(COLUMN_MAP), dtype=DTYPES,
                                 chunksize=chunksize, keep_default_na=False)
            for chunk in reader:
                chunk = chunk.rename(columns=COLUMN_MAP)
                chunk["source_file"] = path.name
                copy_frame(cur, chunk[RAW_COLUMNS], "raw.flights", RAW_COLUMNS)
                conn.commit()
                month_rows += len(chunk)
        total += month_rows
        logger.info("Loaded %s: %s rows (cumulative %s)", path.name, month_rows, total)
    cur.execute("ANALYZE raw.flights")
    conn.commit()
    cur.close()
    return total


def load_airports(conn):
    path = RAW_DIR / "airports.csv"
    if not path.exists():
        download(AIRPORTS_URL, path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["iata_code"].str.fullmatch(r"[A-Z0-9]{3}")]
    df = df.rename(columns={"type": "airport_type"})
    cols = ["iata_code", "ident", "airport_type", "name", "latitude_deg", "longitude_deg",
            "iso_country", "iso_region", "municipality"]
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE raw.airports")
    copy_frame(cur, df[cols], "raw.airports", cols)
    conn.commit()
    cur.close()
    logger.info("Loaded %s airports with an IATA code into raw.airports", len(df))
    return len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=int(os.environ.get("BTS_YEAR", 2024)))
    parser.add_argument("--months", default=os.environ.get("BTS_MONTHS", "1,2,3,4"))
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    months = [int(m) for m in args.months.split(",")]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    conn = get_conn()
    try:
        n_flights = load_flights(conn, args.year, months, args.chunksize)
        n_airports = load_airports(conn)
    finally:
        conn.close()
    print(f"Loaded {n_flights} flights ({args.year} months {months}) into raw.flights and "
          f"{n_airports} airports into raw.airports in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
