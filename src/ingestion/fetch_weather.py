"""
Hourly airport weather (origin + destination features) and airport time zones.

Source: Open-Meteo historical weather API (free, no key):
  https://archive-api.open-meteo.com/v1/archive
Hourly variables: temperature_2m, precipitation, snowfall, wind_speed_10m,
wind_gusts_10m, cloud_cover_low (the archive has no visibility; low cloud cover is
the closest ceiling proxy). Times are requested in UTC (timezone=GMT). Precipitation /
snowfall at hour T are the sums over the hour ending at T; the others are values at T,
so the row for hour T is fully known at time T.

Airport time zones come from the same API (timezone=auto returns the IANA zone of
the coordinates). BTS times are local clock times, so every timing rule of the
2-hours-before-departure features is evaluated in UTC with these zones.

Coordinates: OurAirports (raw.airports), matched on IATA code with the US ICAO
fallback ('K' + code), exactly like dim_airport.

Caching / politeness:
  * data/raw/weather/airport_timezones.csv and data/raw/weather/weather_{YYYY}_{MM}.csv
    are the cache; a month is only requested for airports missing from its file.
  * up to 50 airports per request, one request per batch per month (~7 requests per
    month for ~340 airports), a pause between requests and back-off on HTTP 429.
  * each month file covers the month plus one day on each side (UTC), because early-
    morning cutoffs in the Pacific fall on the previous UTC day.
Then the months are upserted into raw.weather_hourly (PRIMARY KEY airport, hour).

Usage: python fetch_weather.py                # every month present in raw.flights
       python fetch_weather.py --months 2024-04
"""
import argparse
import io
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
import warnings

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
CACHE_DIR = RAW_DIR / "weather"
API = "https://archive-api.open-meteo.com/v1/archive"
BATCH = 50
PAUSE_SECONDS = float(os.environ.get("WEATHER_REQUEST_PAUSE", "12"))
HOURLY = {  # API variable -> raw.weather_hourly column
    "temperature_2m": "temperature_c",
    "precipitation": "precipitation_mm",
    "snowfall": "snowfall_cm",
    "wind_speed_10m": "wind_speed_kmh",
    "wind_gusts_10m": "wind_gusts_kmh",
    "cloud_cover_low": "cloud_cover_low_pct",
}
WX_COLS = ["airport_code", "ts_utc"] + list(HOURLY.values())

DDL = """
CREATE TABLE IF NOT EXISTS raw.airport_timezones (
    airport_code TEXT PRIMARY KEY, timezone TEXT NOT NULL, latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION, loaded_at TIMESTAMP DEFAULT NOW());
CREATE TABLE IF NOT EXISTS raw.weather_hourly (
    airport_code TEXT NOT NULL, ts_utc TIMESTAMP NOT NULL, temperature_c DOUBLE PRECISION,
    precipitation_mm DOUBLE PRECISION, snowfall_cm DOUBLE PRECISION, wind_speed_kmh DOUBLE PRECISION,
    wind_gusts_kmh DOUBLE PRECISION, cloud_cover_low_pct DOUBLE PRECISION,
    loaded_at TIMESTAMP DEFAULT NOW(), PRIMARY KEY (airport_code, ts_utc));
"""

# Airports used by the flights, with coordinates (IATA match, else US ICAO 'K' + code).
AIRPORTS_SQL = """
WITH codes AS (
    SELECT origin AS code FROM raw.flights UNION SELECT dest FROM raw.flights
), ranked AS (
    SELECT iata_code, ident, latitude_deg, longitude_deg,
           row_number() OVER (PARTITION BY iata_code ORDER BY (iso_country = 'US') DESC,
               CASE airport_type WHEN 'large_airport' THEN 1 WHEN 'medium_airport' THEN 2
                                 WHEN 'small_airport' THEN 3 ELSE 4 END, ident) AS rn
    FROM raw.airports
)
SELECT c.code AS airport_code,
       coalesce(a.latitude_deg, k.latitude_deg)   AS latitude,
       coalesce(a.longitude_deg, k.longitude_deg) AS longitude
FROM codes c
LEFT JOIN ranked a ON a.iata_code = c.code AND a.rn = 1
LEFT JOIN raw.airports k ON k.ident = 'K' || c.code AND a.iata_code IS NULL
ORDER BY c.code
"""


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5436"),
        user=os.environ.get("POSTGRES_USER", "flight_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "flight_pass"),
        dbname=os.environ.get("POSTGRES_DB", "flight_delay"),
    )


def api_get(params):
    """One polite request; retries with back-off on 429 / transient errors."""
    url = f"{API}?{urllib.parse.urlencode(params, safe=',')}"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "portfolio flight-delay pipeline"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            time.sleep(PAUSE_SECONDS)
            return data if isinstance(data, list) else [data]
        except urllib.error.HTTPError as e:
            wait = 65 if e.code == 429 else 20 * (attempt + 1)
            logger.warning("HTTP %s from Open-Meteo (attempt %s), waiting %ss: %s",
                           e.code, attempt + 1, wait, e.read()[:200])
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 20 * (attempt + 1)
            logger.warning("Network error %s (attempt %s), waiting %ss", e, attempt + 1, wait)
        time.sleep(wait)
    raise RuntimeError("Open-Meteo request failed after retries")


def batches(df):
    for i in range(0, len(df), BATCH):
        yield df.iloc[i:i + BATCH]


def coords(b):
    return {"latitude": ",".join(f"{x:.4f}" for x in b["latitude"]),
            "longitude": ",".join(f"{x:.4f}" for x in b["longitude"])}


def ensure_timezones(airports):
    path = CACHE_DIR / "airport_timezones.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["airport_code", "timezone"])
    missing = airports[~airports["airport_code"].isin(cached["airport_code"])]
    if len(missing):
        logger.info("Looking up time zones for %s airports", len(missing))
        rows = []
        for b in batches(missing):
            res = api_get({**coords(b), "start_date": "2024-01-01", "end_date": "2024-01-01",
                           "hourly": "temperature_2m", "timezone": "auto"})
            assert len(res) == len(b)
            rows += [{"airport_code": code, "timezone": r["timezone"]} for code, r in zip(b["airport_code"], res)]
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).sort_values("airport_code")
        cached.to_csv(path, index=False)
    return airports.merge(cached, on="airport_code", how="left")


def month_range(year, month):
    first = date(year, month, 1)
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return first - timedelta(days=1), nxt  # one extra day on each side (UTC)


def ensure_month(airports, year, month):
    path = CACHE_DIR / f"weather_{year}_{month:02d}.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=WX_COLS)
    missing = airports[~airports["airport_code"].isin(cached["airport_code"].unique())]
    if len(missing):
        start, end = month_range(year, month)
        logger.info("Downloading %s-%02d weather for %s airports (%s requests)", year, month,
                    len(missing), -(-len(missing) // BATCH))
        frames = []
        for b in batches(missing):
            res = api_get({**coords(b), "start_date": start.isoformat(), "end_date": end.isoformat(),
                           "hourly": ",".join(HOURLY), "timezone": "GMT"})
            assert len(res) == len(b)
            for code, r in zip(b["airport_code"], res):
                h = pd.DataFrame(r["hourly"]).rename(columns={"time": "ts_utc", **HOURLY})
                h.insert(0, "airport_code", code)
                frames.append(h[WX_COLS])
        cached = pd.concat(([cached] if len(cached) else []) + frames, ignore_index=True)
        cached = cached.sort_values(["airport_code", "ts_utc"]).reset_index(drop=True)
        cached.to_csv(path, index=False)
    else:
        logger.info("%s-%02d weather: all %s airports already cached in %s", year, month, len(airports), path.name)
    return cached


def upsert(conn, table, df, key, cols):
    cur = conn.cursor()
    cur.execute(f"CREATE TEMP TABLE tmp_up (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
    buf = io.StringIO()
    df[cols].to_csv(buf, index=False, header=False)
    buf.seek(0)
    cur.copy_expert(f"COPY tmp_up ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '')", buf)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key) + ", loaded_at = NOW()"
    cur.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT {', '.join(cols)} FROM tmp_up "
                f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {updates}")
    n = cur.rowcount
    conn.commit()
    cur.close()
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="", help="comma-separated YYYY-MM; default: all months in raw.flights")
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(DDL)
        conn.commit()
        airports = pd.read_sql(AIRPORTS_SQL, conn)
        no_xy = airports[airports["latitude"].isna()]
        if len(no_xy):
            raise ValueError(f"airports without coordinates: {no_xy['airport_code'].tolist()}")
        if args.months:
            months = [tuple(int(x) for x in m.split("-")) for m in args.months.split(",")]
        else:
            months = [tuple(r) for r in pd.read_sql(
                "SELECT DISTINCT year::int AS y, month::int AS m FROM raw.flights ORDER BY 1, 2", conn).to_numpy()]
        airports = ensure_timezones(airports)
        n_tz = upsert(conn, "raw.airport_timezones", airports, ["airport_code"],
                      ["airport_code", "timezone", "latitude", "longitude"])
        logger.info("Upserted %s airport time zones (%s distinct zones)", n_tz, airports["timezone"].nunique())
        for y, m in months:
            wx = ensure_month(airports, y, m)
            wx = wx[wx["airport_code"].isin(airports["airport_code"])]
            start, end = month_range(y, m)
            cur.execute("SELECT count(*) FROM raw.weather_hourly WHERE ts_utc >= %s AND ts_utc < %s + 1 "
                        "AND airport_code = ANY(%s)", (start, end, list(airports["airport_code"])))
            if cur.fetchone()[0] >= len(wx):
                logger.info("%s-%02d weather already in raw.weather_hourly (%s rows), skipping upsert", y, m, len(wx))
                continue
            n = upsert(conn, "raw.weather_hourly", wx, ["airport_code", "ts_utc"], WX_COLS)
            logger.info("Upserted %s hourly weather rows for %s-%02d", n, y, m)
        cur.execute("ANALYZE raw.weather_hourly")
        conn.commit()
        cur.execute("SELECT count(*), count(DISTINCT airport_code), min(ts_utc), max(ts_utc) FROM raw.weather_hourly")
        print("raw.weather_hourly rows / airports / first hour / last hour:", cur.fetchone())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
