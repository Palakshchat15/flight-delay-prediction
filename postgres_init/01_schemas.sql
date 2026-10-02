-- Flight Delay Analytics: schema layout (raw -> staging -> intermediate -> warehouse)
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS warehouse;

-- Raw landing table for BTS Reporting Carrier On-Time Performance monthly files.
-- Only the columns the project needs are kept (the source has 110). Post-flight
-- columns (actual times, delays, delay causes) are kept for the descriptive
-- dashboards but are NEVER used as model features (see README_PIPELINE.md).
-- Times are kept as the source's hhmm text so no information is lost at landing.
CREATE TABLE IF NOT EXISTS raw.flights (
    flight_date          DATE,
    year                 SMALLINT,
    month                SMALLINT,
    day_of_month         SMALLINT,
    day_of_week          SMALLINT,
    reporting_airline    TEXT,
    tail_number          TEXT,
    flight_number        TEXT,
    origin               TEXT,
    origin_city_name     TEXT,
    origin_state         TEXT,
    dest                 TEXT,
    dest_city_name       TEXT,
    dest_state           TEXT,
    crs_dep_time         TEXT,
    dep_time             TEXT,
    dep_delay            NUMERIC,
    crs_arr_time         TEXT,
    arr_time             TEXT,
    arr_delay            NUMERIC,
    arr_del15            NUMERIC,
    cancelled            NUMERIC,
    cancellation_code    TEXT,
    diverted             NUMERIC,
    crs_elapsed_time     NUMERIC,
    distance             NUMERIC,
    carrier_delay        NUMERIC,
    weather_delay        NUMERIC,
    nas_delay            NUMERIC,
    security_delay       NUMERIC,
    late_aircraft_delay  NUMERIC,
    source_file          TEXT,
    loaded_at            TIMESTAMP DEFAULT NOW()
);

-- OurAirports reference data (only airports with an IATA code).
CREATE TABLE IF NOT EXISTS raw.airports (
    iata_code      TEXT,
    ident          TEXT,
    airport_type   TEXT,
    name           TEXT,
    latitude_deg   DOUBLE PRECISION,
    longitude_deg  DOUBLE PRECISION,
    iso_country    TEXT,
    iso_region     TEXT,
    municipality   TEXT,
    loaded_at      TIMESTAMP DEFAULT NOW()
);

-- IANA time zone per airport (from Open-Meteo's timezone=auto lookup on the airport
-- coordinates). Needed because BTS times are LOCAL clock times: the 2-hours-before-
-- departure cutoff and the inbound aircraft's timeline are compared in UTC.
CREATE TABLE IF NOT EXISTS raw.airport_timezones (
    airport_code   TEXT PRIMARY KEY,
    timezone       TEXT NOT NULL,
    latitude       DOUBLE PRECISION,
    longitude      DOUBLE PRECISION,
    loaded_at      TIMESTAMP DEFAULT NOW()
);

-- Hourly weather per airport (Open-Meteo historical archive, UTC hours). Observed
-- (reanalysis) weather; for the destination it stands in for a forecast (README).
CREATE TABLE IF NOT EXISTS raw.weather_hourly (
    airport_code     TEXT NOT NULL,
    ts_utc           TIMESTAMP NOT NULL,
    temperature_c    DOUBLE PRECISION,
    precipitation_mm DOUBLE PRECISION,
    snowfall_cm      DOUBLE PRECISION,
    wind_speed_kmh   DOUBLE PRECISION,
    wind_gusts_kmh   DOUBLE PRECISION,
    cloud_cover_low_pct DOUBLE PRECISION,
    loaded_at        TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (airport_code, ts_utc)
);
