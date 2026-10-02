# US Airline Flight Delay Analytics and Prediction Pipeline

End-to-end data engineering + data science portfolio project on **real US flight data**:
the Bureau of Transportation Statistics (BTS) *Reporting Carrier On-Time Performance*
files for **January to April 2024 (2,240,444 scheduled flights)**, plus **hourly airport weather**
from the Open-Meteo historical archive. The pipeline lands the raw files in Postgres, validates
them with Great Expectations, models them into a star schema with dbt, builds features that are
known **2 hours before scheduled departure** (with an automated leakage check), trains a
flight-delay classifier with a strictly time-ordered validation/test design, and ends in two
dashboards: an **Excel dashboard** and a **Tableau workbook**.

Every number below comes from the verified run on 2026-09-26 described in
"Verified run evidence" at the end.

## The prediction question changed: "weeks ahead" -> "2 hours before departure"

The first version of this project predicted delays from information known **weeks ahead**
(the schedule and last month's delay rates). On the unseen April 2024 test month the models
reached only AUC-ROC 0.65 / AUC-PR 0.29-0.30, because most delays come from same-day events.
The question is now: **"2 hours before this flight's scheduled departure, will it arrive 15+
minutes late?"** At that moment an airline already knows where the aircraft is (the inbound
flight), how the airports and its own network are running, and the weather. Features may use
anything known at that moment and nothing later; each timing rule is documented below and
enforced by an automated leakage check.

**The old and new numbers answer different questions and are not directly comparable.** A
2-hours-before model *should* score higher; the useful comparison is how much it adds over the
obvious same-day rule ("the inbound aircraft is already late"), which is included as a baseline.

### Before -> after (April 2024 test month, 576,915 completed flights, 19.06% delayed)

| Model | Question | AUC-ROC | AUC-PR | Precision @20% | Recall @20% | Precision @10% | Recall @10% |
|---|---|---|---|---|---|---|---|
| Historical carrier+route rule | weeks ahead (before) | 0.5937 | 0.2578 | 0.2688 | 0.2821 | 0.2992 | 0.1570 |
| Logistic regression | weeks ahead (before) | 0.6538 | 0.2969 | 0.3197 | 0.3355 | n/a | n/a |
| LightGBM | weeks ahead (before) | 0.6492 | 0.2929 | 0.3205 | 0.3363 | 0.3520 | 0.1847 |
| **Inbound-delay rule** (new baseline) | 2 h before (after) | 0.6477 | 0.3149 | 0.3769 | 0.3955 | 0.4735 | 0.2485 |
| Logistic regression | 2 h before (after) | 0.7642 | 0.5104 | 0.4839 | 0.5078 | 0.6474 | 0.3397 |
| **LightGBM** | 2 h before (after) | **0.8088** | **0.6330** | **0.5489** | **0.5760** | **0.7647** | **0.4013** |

"@20%" / "@10%": every model flags its riskiest 20% (10%) of the April flights, so precision and
recall are directly comparable across models. The historical rule is identical under both
questions (it uses no same-day information). The "before" rows are the previous README's April
results; the weeks-ahead LightGBM was re-fit in this run with the same code, split and settings
and reproduced them exactly (AUC-ROC 0.6492, AUC-PR 0.2929), which is where its 10% numbers come
from. The previous run did not report 10% numbers for logistic regression (n/a). A random ranking
scores AUC-PR 0.19 (the April delay rate).

## Results in one paragraph

Two hours before departure, **LightGBM ranks April flights with AUC-ROC 0.809 and AUC-PR 0.633**.
When it flags the riskiest 20% of flights, **55% of the flagged flights are actually delayed and they
cover 58% of all delayed flights**. When it flags the riskiest 10%, 76% are delayed (40% of all delays).
The obvious same-day rule, which ranks flights by how late the inbound aircraft already is, gets
precision 38% / recall 40% at the same 20% flag rate, so the model adds about 17 points of precision
and 18 points of recall on top of that rule. LightGBM also clearly beats logistic regression here (+0.045
AUC-ROC, +0.123 AUC-PR; day-level bootstrap 95% intervals exclude zero). Its probabilities are close
to calibrated (mean 19.7% predicted vs 19.1% actual; top decile 77.6% vs 76.5%). Main caveat: BTS tail
numbers record the aircraft that actually flew, so same-day aircraft swaps are visible after the fact.
On the 98% of April flights without a swap signature the model scores AUC-ROC 0.796 / AUC-PR 0.594,
a bit lower than on all flights (see "Residual leakage risk").

## Architecture

```
transtats.bts.gov monthly zips (2024_1 .. 2024_4)    OurAirports airports.csv    Open-Meteo archive
        |                                                 |                       (hourly weather +
        v  src/ingestion/load_raw_flights.py              |                        IANA time zones)
Postgres raw.flights (2,240,444 rows)            raw.airports (9,054)                  |
        |                                                 +---> src/ingestion/fetch_weather.py
        |                                                       (cached in data/raw/weather/)
        |                                          raw.airport_timezones (335), raw.weather_hourly (988,920)
        v  great_expectations/validate_raw_flights.py   (22 hard gates + 2 warn-only)
        |
        v  dbt seed + dbt run (dbt/flight_dbt)
staging.stg_flights, stg_airports, carriers (seed)
        |
        v
intermediate.int_flights_enriched  (status, schedule context, UTC timeline: sched/actual dep+arr, cutoff)
        |
        v
warehouse: dim_carrier, dim_airport, fact_flights, mart_flight_features, dashboard marts
        |
        v  src/ml_pipeline/cutoff_features.py       -> warehouse.flight_cutoff_features (34 features +
        |                                              event timestamps of everything each feature used)
        v  src/ml_pipeline/leakage_check.py         (timestamps < cutoff, SQL key check, independent recompute)
        |
        v  dbt test (66 tests, incl. 3 leakage / time-zone singular tests)
        |
        v  src/ml_pipeline/train_delay_model.py
outputs/model_metrics.json, model_report.txt, shap_summary.png, leakage_check.json,
outputs/predicted_delays_test_month.csv, outputs/model/lgbm_delay_model.txt, warehouse.model_test_predictions
        |
        v  src/dashboards/export_dashboard_data.py  -> data/processed/*.csv (+ cross-checks)
        +--> src/dashboards/build_excel_dashboard.py  -> outputs/Flight_Delays_Dashboard.xlsx
        +--> src/dashboards/build_tableau_workbook.py -> tableau/Flight_Delays.twbx
```

Airflow DAG `flight_delay_pipeline_dag` (`airflow/dags/flight_delay_pipeline_dag.py`, triggered
manually, `schedule_interval=None`):

```
wait_for_raw_dir (PythonSensor) -> ingest_raw_flights -> fetch_weather -> great_expectations_validate
  -> dbt_run (seed + run) -> build_cutoff_features -> leakage_check -> dbt_test -> train_delay_model
  -> export_dashboard_data -> [build_excel_dashboard, build_tableau_workbook]
```

`src/ml_pipeline/design_checks.py` (validation-month experiments, about 7 minutes) is run by hand,
not by the DAG; its output is `outputs/design_checks.json`. No GenAI step is part of this project.

## Folder structure

```
airflow/dags/            DAG definition (mounted into the containers)
airflow/Dockerfile       apache/airflow:2.9.3-python3.11 + requirements-airflow.txt
dbt/profiles.yml         env-var driven Postgres profile
dbt/flight_dbt/          dbt project: seeds, staging -> intermediate -> marts, macros, tests/ (singular tests)
data/raw/                BTS zips + airports.csv + weather/ cache (git-ignored; downloaded if missing)
data/processed/          small dashboard CSVs (regenerated every run)
great_expectations/      validate_raw_flights.py (GE 0.18 in-memory suite, hard gates + warnings)
postgres_init/           schemas + raw table DDL (flights, airports, airport_timezones, weather_hourly)
src/ingestion/           load_raw_flights.py, fetch_weather.py
src/ml_pipeline/         cutoff_features.py, leakage_check.py, train_delay_model.py, design_checks.py, db_utils.py
src/dashboards/          export_dashboard_data.py, build_excel_dashboard.py, build_tableau_workbook.py
outputs/                 metrics, report, design checks, leakage check, SHAP plot, predictions, model, Excel
outputs/previous_weeks_ahead/  metrics + report of the previous "weeks ahead" model (for the comparison)
tableau/                 Flight_Delays.twbx (extract-based, opens in Tableau Public)
docker-compose.yml       postgres (5436), airflow-webserver (8085), airflow-scheduler, init jobs
```

## Stack

Airflow 2.9.3 (LocalExecutor, Python 3.11) - PostgreSQL 15 - dbt-core/dbt-postgres 1.7.13 -
Great Expectations 0.18.19 - pandas 2.1.4 / numpy 1.26.4 - scikit-learn 1.4.2 - LightGBM 4.3.0 -
SHAP 0.45.0 - openpyxl 3.1.2 - tableauhyperapi 0.0.26700. `sqlalchemy>=1.4.36,<2.0` is pinned
because Airflow 2.9.3 and dbt-postgres 1.7.13 both need SQLAlchemy 1.4.

## How to run

```bash
cd flight_delay_project
cp .env.example .env          # set HOST_PROJECT_DIR to this folder (forward slashes)
docker compose up -d --build  # postgres :5436, Airflow UI http://localhost:8085 (admin/admin)
docker compose exec airflow-scheduler airflow dags list-import-errors
docker compose exec airflow-scheduler airflow dags trigger flight_delay_pipeline_dag
docker compose exec airflow-scheduler airflow dags list-runs -d flight_delay_pipeline_dag
docker compose exec airflow-scheduler airflow tasks states-for-dag-run flight_delay_pipeline_dag <run_id>
# optional, validation-month design experiments (not part of the DAG):
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/src/ml_pipeline && python design_checks.py"
```

The raw zips and the weather do not need to be downloaded by hand: the ingest task downloads any
missing BTS month (months set by `BTS_YEAR` / `BTS_MONTHS` in `.env`) and `fetch_weather` downloads
any missing weather month (then reuses the cache). The model uses the last four months: the first is
history only, the second-to-last is the validation month and the last is the test month. A full DAG
run took about 20 minutes on a 12-core laptop (Docker VM with 7.6 GB RAM, shared with another
project's stack), most of it model training, SHAP and the per-flight reasons.

Postgres note: the leakage check's four-way join over 2.2M rows ran out of Docker's `/dev/shm` with
parallel hash joins, so that query runs with `max_parallel_workers_per_gather = 0`. The training data
is streamed in 200k-row chunks (server-side cursor) and held as float32: a one-shot fetch was
OOM-killed in the shared 7.6 GB VM.

### Runs on an 8 GB laptop (Docker VM capped at 4 GB)

On an 8 GB Windows laptop, Docker's WSL2 VM gets about 4 GB by default. The pipeline was tested
under exactly that cap (`memory=4GB` in `.wslconfig`, 3.8 GB visible to Docker):

| | Before | After |
|---|---|---|
| Full DAG run | `great_expectations_validate` OOM-killed (process ~2.6 GB) | all 12 tasks succeed, ~25 min |
| Peak memory (all containers) | killed | ~3.1 GB of 3.8 GB, 0 kernel OOM kills |
| Outputs | - | byte-identical to the previous run |

- **Unchanged:** the pipeline logic. The DAG file, every expectation, the dbt models and tests,
  the features, the splits, the models and the metrics are all the same.
- **Proof of identical outputs:** model_metrics.json, model_report.txt, the April predictions,
  feature importance, leakage_check.json and design_checks match the pre-change run byte for byte.
  The dashboard CSVs match except for line endings (written in the Linux container, not on Windows).
- **What changed is only how memory is used:**
  - `great_expectations/validate_raw_flights.py`: the same expectations, run on blocks of flight
    dates and combined. A check passes only if it passes on every block. The duplicate-flight key
    includes `flight_date`, so the uniqueness check stays exact.
  - `src/ml_pipeline/db_utils.py`: new `read_sql_chunks` (server-side cursor, 200k-row chunks).
  - `cutoff_features.py` and `train_delay_model.py`: stream the same rows in the same order
    instead of one large `read_sql`. `flight_key` comes first via COPY, then the other columns
    in groups.
  - `docker-compose.yml`: `AIRFLOW__CORE__PARALLELISM=2` (the DAG never runs more than two tasks
    at once), 2 web workers, `shared_buffers=256MB`, and glibc/Python allocator settings that
    return freed memory to the OS.
- **Webserver start-up:** if Docker restarts in the middle of a run, the scheduler resumes the
  heavy task straight away, and the webserver can miss its 120-second start-up window. It is
  harmless: restart it with `docker compose up -d airflow-webserver`. For demos, start the stack
  first and trigger the DAG afterwards.

Open the outputs: `outputs/Flight_Delays_Dashboard.xlsx` in Excel, `tableau/Flight_Delays.twbx`
in Tableau Public (or Desktop).

## Data source

- **Flights**: BTS "Reporting Carrier On-Time Performance (1987-present)", downloaded directly
  from `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_{1..4}.zip`
  (no account needed). Each zip holds one ~250 MB CSV with 110 columns; the loader checks that every
  file is a real zip containing exactly one CSV and keeps 31 columns. Rows per month: Jan 547,271,
  Feb 519,221, Mar 591,767, Apr 582,185. 15 reporting carriers.
- **Airports**: OurAirports `https://davidmegginson.github.io/ourairports-data/airports.csv`
  (public domain), joined on IATA code for names and coordinates. All 335 airports in the data
  got coordinates. One needed a fallback: OurAirports now lists Palm Beach (PBI) under IATA
  `DJT` after its renaming, so `dim_airport` falls back to the US ICAO ident (`'K' || code`).
- **Weather + time zones**: Open-Meteo historical weather API
  `https://archive-api.open-meteo.com/v1/archive` (free, no key), hourly `temperature_2m`,
  `precipitation`, `snowfall`, `wind_speed_10m`, `wind_gusts_10m`, `cloud_cover_low` in UTC, for the 335
  airports' OurAirports coordinates; the same API with `timezone=auto` gives each airport's IANA zone.
  `src/ingestion/fetch_weather.py` batches 50 airports per request (one request per batch per month, a
  12 s pause between requests, back-off on HTTP 429), caches `data/raw/weather/weather_YYYY_MM.csv` and
  `airport_timezones.csv`, and upserts into `raw.weather_hourly` / `raw.airport_timezones`; later runs
  only request airports or months missing from the cache and skip months already in the table. Each
  month file covers one extra UTC day on each side (Pacific early-morning cutoffs fall on the previous
  UTC day). The archive is reanalysis (observed) weather, which for the destination stands in for a
  forecast (see Limitations).

## Cleaning and business-rule decisions

| Topic | Decision |
|---|---|
| Cancelled flights (32,546, 1.45%) | Real business events, not errors. Kept in `fact_flights` and drive the **cancellation-rate KPI** (share of scheduled flights). Excluded from the delay model because they never arrive. |
| Diverted flights (4,931, 0.22%) | Kept in the fact table, excluded from on-time % and from the model (no arrival at the scheduled destination, so no ArrDel15). |
| Missing arrival delay | GE checks that it is missing **only** for cancelled/diverted flights (hard gate: every completed flight has `arr_delay` and `arr_del15`, and `arr_del15` equals `arr_delay >= 15`). All passed. |
| On-time % definition | Completed flights arriving less than 15 minutes late, divided by completed (not cancelled, not diverted) flights. |
| `2400` scheduled time | BTS writes end-of-day midnight as `2400` (1 scheduled departure in Jan-Apr 2024). GE allows it; staging maps it to `2359` so it stays on the same date and hour. This was found by the GE gate failing on the first run. |
| Overnight arrivals | BTS gives local hhmm times but no arrival date. 67,772 flights (3.0%) are scheduled to land after local midnight. `int_flights_enriched.sched_arr_date` recovers the day offset as `round((dep_minute + scheduled_block - arr_minute) / 1440)` (time-zone differences within the US, under 12 h, cannot flip the rounding; the one flight without a block time is assumed same-day), and destination volume per hour is counted on that arrival date. |
| Duplicates | Natural key date + carrier + flight number + origin + scheduled departure. GE hard-gates on uniqueness (0 duplicates found); staging also dedupes with `row_number()` defensively. |
| Carrier names | `dbt/flight_dbt/seeds/carriers.csv` (code to name for the 15 reporting carriers). |

Great Expectations hard gates: key columns not null, 3-character IATA airport codes, valid hhmm
scheduled times, cancelled/diverted are 0/1 flags, distance 10-6,000 miles, no duplicate
flights, completed flights have arrival delay and a consistent ArrDel15, each monthly file has
400k-800k rows. Warn-only: overall cancellation rate between 0.5% and 8% (observed 1.45%),
arrival delay within [-120, 2000] minutes for 99.99% of completed flights (41 rows outside, kept).

## Model: 2 hours before departure, will a flight arrive 15+ minutes late?

**Label**: BTS `ArrDel15` on completed flights (2,202,967 flights; 19.99% delayed overall).
Monthly delay rates swing a lot: Jan 24.06% (winter storms), Feb 15.92%, Mar 20.85%, Apr 19.06%.

**Prediction point**: `cutoff = scheduled departure - 2 hours`. BTS times are local clock times,
so every timing comparison is done in UTC: each airport's IANA time zone (from Open-Meteo, 22
zones) converts scheduled departures with Postgres `AT TIME ZONE` (DST-aware), and
`sched_arr_utc = sched_dep_utc + scheduled block time`. Actual event times are
`actual_dep = sched_dep + DepDelay` and `actual_arr = sched_arr + ArrDelay`. A flight "has departed
by the cutoff" only if `actual_dep < cutoff`, "has arrived" only if `actual_arr < cutoff`.
Time-zone sanity check: the block time implied by the local clock times and zones matches BTS's
scheduled elapsed time within 5 minutes for 99.985% of flights (dbt test + leakage check).

### Time-based split with a validation month (last four months of data)

| Month | Flights | Delayed | Role |
|---|---|---|---|
| Jan 2024 | 525,370 | 24.06% | History only: feeds February's previous-month rate features; never fitted on |
| Feb 2024 | 515,269 | 15.92% | Stage 1 fit |
| Mar 2024 | 585,413 | 20.85% | **Validation**: feature groups, the inbound-rule threshold and the model comparison are judged here |
| Feb-Mar 2024 | 1,100,682 | 18.54% | Final refit with the choices frozen |
| Apr 2024 | 576,915 | 19.06% | **Test**: scored once by the refit models |

A random split would put flights from the same day, and the same storm, into both train and test.
`design_checks.py` excludes the test month in SQL and never reads it.

### Features and their timing rules

All same-day features are built by `src/ml_pipeline/cutoff_features.py` into
`warehouse.flight_cutoff_features`, which also stores, for every row, the time of the latest event
each feature group used (`*_ts`) and the keys of the legs used, so the rules can be checked.

**Schedule + history (the old weeks-ahead set, 19 features, unchanged)**: scheduled hours, departure
minute, day of week, block time, distance, scheduled airport volume (published schedule), carrier;
previous-calendar-month delay rates as lift for carrier, origin, destination, route, carrier+route,
origin+hour, carrier+hour; previous-month median rotation per scheduled flight. Legitimate because
they come from the schedule or from a month that ended before the scored month began.

**Inbound aircraft (12 features).** "Previous leg" = the same `Tail_Number`'s flight with the latest
scheduled departure strictly before this flight's, within the previous 24 h (cancelled legs
included, since they are part of the published rotation).

| Feature | Value | Timing rule that makes it legitimate |
|---|---|---|
| `inb_has_prev` | previous leg exists | schedule |
| `inb_sched_turn_min` | this sched. departure - previous sched. arrival | schedule (see residual risk below) |
| `inb_prev_sched_dep_before_cutoff` | previous leg was scheduled to leave before the cutoff | schedule |
| `inb_prev_departed` | 1 if previous `actual_dep < cutoff` | a departure is observable once it happens |
| `inb_prev_dep_delay` | previous leg's departure delay, **only if** it departed before the cutoff, else NULL | known at the previous leg's actual departure |
| `inb_prev_arrived` | 1 if previous `actual_arr < cutoff` | observable once it happens |
| `inb_prev_arr_delay` | previous leg's arrival delay, **only if** it arrived before the cutoff, else NULL | known at the previous leg's actual arrival |
| `inb_prev_overdue_min` | if scheduled to leave before the cutoff but has **not** left by it: cutoff - its sched. departure, else 0 | "hasn't departed yet although scheduled to" is visible at the cutoff |
| `inb_known_delay_min` | arrival delay if arrived; else max(departure delay, cutoff - sched. arrival) if departed; else the overdue minutes; else 0 | combination of the rules above |
| `inb_projected_slack_min` | `inb_sched_turn_min - max(inb_known_delay_min, 0)` | combination of the rules above |
| `ac_latest_dep_delay` | departure delay of the aircraft's most recent leg with `actual_dep` in [cutoff - 24 h, cutoff) | known at that departure |
| `ac_latest_arr_delay` | arrival delay of its most recent leg with `actual_arr` in [cutoff - 24 h, cutoff) | known at that arrival |

On completed Jan-Apr flights: 97.2% have a previous leg; that leg had departed by the cutoff for
86.8% and arrived for 27.6% (a typical rotation is block time + 45-60 min turn, so the inbound is
often still in the air 2 hours before departure).

**Congestion in the 2 hours before the cutoff (9 features)**, window `[cutoff - 2 h, cutoff)`:
origin departure count and mean departure delay (departures with `actual_dep` in the window);
origin flights scheduled to leave in the window that have **not** departed by the cutoff (count and
share; cancelled flights count as not departed); mean arrival delay of arrivals into the origin and
into the destination, mean departure delay at the destination, the carrier's network-wide mean
departure delay, and the whole network's mean departure delay (all with the event time in the window;
delays clipped to [-30, 300] min; NULL for an empty window).

**Weather (13 features)**, Open-Meteo hourly (temperature, precipitation, snowfall, wind speed, gusts,
low cloud cover; UTC hours; hour T holds values at T and sums over the hour ending at T):
- origin at the last full hour at or before the cutoff, plus precipitation over the 3 hours ending then
  (known at the cutoff);
- destination at the scheduled arrival hour. **This is observed weather standing in for a forecast**:
  a real system would use the forecast issued at the cutoff, which is less accurate, so these features
  are somewhat optimistic (limitation below). On validation they add +0.006 AUC-ROC / +0.005 AUC-PR.

**Never used**: this flight's own `DepTime/DepDelay/DepDel15`, `TaxiOut`, `WheelsOff/WheelsOn`, `TaxiIn`,
actual times, `ArrDelay` (label), elapsed/air time, delay-cause minutes, cancellation/diversion
outcomes, and the same-day tail-number *schedule* features from the weeks-ahead version
(`aircraft_leg_of_day`, `aircraft_legs_scheduled_today`, `sched_turnaround_minutes`).

### Automated leakage check

`src/ml_pipeline/leakage_check.py` runs as DAG task `leakage_check` (before `dbt_test` and training)
and fails the run on any violation; its report is `outputs/leakage_check.json`. Results of the verified run:

1. **Timestamp rule**: every stored event time (`inb_prev_dep_ts`, `inb_prev_arr_ts`, `ac_latest_*_ts`,
   and the latest event in each congestion window) is strictly before `cutoff_utc` for all 2,202,967
   rows; the origin weather hour is at or before the cutoff; every "only if it already happened" value
   is NULL exactly when its event time is NULL. Violations: **0** (latest event used: 1 minute before
   the cutoff).
2. **Key cross-check in SQL**: re-joins the stored inbound / latest legs to `int_flights_enriched`:
   the inbound departure (arrival) delay is filled exactly when that leg departed (arrived) before the
   cutoff and equals the recorded delay; the previous leg is strictly earlier and on the same tail; no
   flight uses its own events. Violations: **0**.
3. **Independent recomputation**: for 400 seeded random flights, a deliberately naive per-flight
   implementation first truncates the world to what was visible at the cutoff and recomputes all 21
   inbound and congestion features: **0 mismatches in 8,400 values**.
4. **Feature-name rule**: none of the 53 model features is one of the flight's own post-departure
   columns.
5. **Time-zone sanity**: 99.985% of flights have consistent UTC block times (threshold 99%).

The same rules also run as dbt singular tests (`tests/assert_cutoff_features_use_only_pre_cutoff_events.sql`,
`assert_inbound_delay_known_only_after_event.sql`, `assert_utc_block_time_matches_schedule.sql`).

**The check caught a real leak during development.** HNL-GUM crosses the date line; the arrival date
inferred from local clock times (the weeks-ahead version's rule) put its scheduled arrival *a day
before* its departure, so the flight's own arrival looked like it happened before its cutoff and its
own arrival delay leaked into `ac_latest_arr_delay` for 46 flights. The key cross-check flagged it
(`own_flight_used = 46`); the fix is `sched_arr_utc = sched_dep_utc + scheduled block time`, and the
builder now asserts that no flight arrives before its own cutoff. It also flagged 242 "previous legs"
scheduled at the same minute as the flight itself (a data artefact); the previous leg must now be
strictly earlier.

### Residual leakage risk (tail numbers)

BTS `Tail_Number` is the aircraft that **actually** operated the flight. When an airline swaps
aircraft on the day because of a delay, the swapped-in aircraft appears with a "previous leg" whose
scheduled arrival is after (or minutes before) this departure. The weeks-ahead investigation found
these impossible turns 74-80% delayed and removed the same-day tail features for that question. Two
hours before departure most swaps are already decided and the inbound's delay is genuinely known, so
the inbound features are used now, but the identity of "this flight's aircraft" is still taken from
after-the-fact data. Quantified on April:

| April test subset | Flights | Delayed | LightGBM AUC-ROC | LightGBM AUC-PR | LogReg AUC-ROC | Inbound rule AUC-ROC |
|---|---|---|---|---|---|---|
| All completed flights | 576,915 | 19.06% | 0.8088 | 0.6330 | 0.7642 | 0.6477 |
| Excluding swap signature (scheduled turn < 20 min or negative) | 566,547 | 17.93% | 0.7957 | 0.5940 | 0.7621 | 0.6585 |
| Swap-signature flights only | 10,368 | 80.87% | | | | |

So 1.8% of flights carry the signature and part of the headline gain comes from them; on the other
98% the model still scores AUC-ROC 0.796 / AUC-PR 0.594, far above the weeks-ahead 0.649 / 0.293.
Treat the headline as slightly optimistic; a production system would use the airline's planned
tail assignment as of the cutoff.

### Design choices, judged on the validation month only

Feature-group ablation (`outputs/design_checks.json`; fit on February, scored on March, same models
and seeds):

| Variant (fit Feb, score Mar) | Features | LogReg AUC-ROC | LogReg AUC-PR | LightGBM AUC-ROC | LightGBM AUC-PR |
|---|---|---|---|---|---|
| Weeks-ahead features only | 19 | 0.6615 | 0.3398 | 0.6600 | 0.3352 |
| + inbound aircraft | 31 | 0.7472 | 0.5095 | 0.7806 | 0.6169 |
| + inbound + congestion | 40 | 0.7586 | 0.5268 | 0.8052 | 0.6446 |
| + inbound + congestion + origin weather | 47 | 0.7637 | 0.5302 | 0.8129 | 0.6518 |
| **+ destination weather (forecast stand-in) = ALL (used)** | 53 | **0.7695** | **0.5345** | **0.8185** | **0.6570** |
| Weeks-ahead + congestion + weather, **no** inbound aircraft | 41 | 0.7050 | 0.4146 | 0.7335 | 0.4557 |

- Every group improves validation AUC-PR for both models, so all four are kept. The inbound aircraft
  is by far the largest single gain (+0.28 AUC-PR for LightGBM); without it, congestion + weather reach
  0.456.
- **Inbound-delay rule threshold**: "flag if the inbound aircraft's known delay >= X min"; X = **5 min**
  maximises F1 on March (F1 0.416, flags 27.1% of March flights; X = 0 gives 0.408, X = 10 gives 0.414).
- Carried over unchanged from the weeks-ahead design checks (`outputs/design_checks_weeks_ahead.json`):
  previous-calendar-month history window, rates as lift, no class weighting. Hyperparameters were not
  tuned (LightGBM: 500 trees, learning rate 0.05, 63 leaves, 200 min child samples, 0.8 row/column
  subsampling). Logistic regression now uses median imputation **with missing-value indicators**,
  because "not known at the cutoff" (NULL) is itself informative for the inbound features.
- Features are held as float32 (memory); the weeks-ahead reference still reproduces the previous
  results to four decimals.

### Models and operating points

1. **Historical rate rule**: score = previous-month carrier+route delay-rate lift (no fitting).
2. **Inbound-delay rule**: score = the inbound aircraft's known delay at the cutoff (`inb_known_delay_min`,
   no fitting). Also reported as a yes/no rule at the validation threshold (>= 5 min).
3. **Logistic regression**: unweighted, imputed + indicators + standardised, one-hot carrier.
4. **LightGBM**: unweighted binary log-loss, categorical carrier, `deterministic=True`.

**Equal operating points**: each model flags the **riskiest 20%** and, separately, the **riskiest
10%** of the scored month's flights, so precision and recall are directly comparable. In a live system
the flags would come from a fixed probability threshold; ranking a whole month is the offline
evaluation convention used throughout this project (ties, e.g. the many flights with inbound known
delay 0, are broken by row order).

### Results

Validation month (March; stage-1 models fitted on February):

| Model | AUC-ROC | AUC-PR | P @20% | R @20% | F1 @20% | P @10% | R @10% |
|---|---|---|---|---|---|---|---|
| Historical rate rule (carrier+route) | 0.6088 | 0.2949 | 0.3100 | 0.2974 | 0.3035 | 0.3465 | 0.1662 |
| Inbound-delay rule | 0.6563 | 0.3416 | 0.4162 | 0.3993 | 0.4076 | 0.5058 | 0.2426 |
| Logistic regression | 0.7695 | 0.5345 | 0.5254 | 0.5041 | 0.5145 | 0.6777 | 0.3251 |
| LightGBM | 0.8190 | 0.6575 | 0.5944 | 0.5703 | 0.5821 | 0.8009 | 0.3842 |

**Test month (April; refit on Feb-Mar; scored once):**

| Model | AUC-ROC | AUC-PR | P @20% | R @20% | F1 @20% | P @10% | R @10% |
|---|---|---|---|---|---|---|---|
| Historical rate rule (carrier+route) | 0.5937 | 0.2578 | 0.2688 | 0.2821 | 0.2753 | 0.2992 | 0.1570 |
| Inbound-delay rule | 0.6477 | 0.3149 | 0.3769 | 0.3955 | 0.3860 | 0.4735 | 0.2485 |
| Logistic regression | 0.7642 | 0.5104 | 0.4839 | 0.5078 | 0.4956 | 0.6474 | 0.3397 |
| **LightGBM** | **0.8088** | **0.6330** | **0.5489** | **0.5760** | **0.5621** | **0.7647** | **0.4013** |
| *LightGBM, weeks-ahead features only (reference)* | 0.6492 | 0.2929 | 0.3205 | 0.3363 | 0.3282 | 0.3520 | 0.1847 |

- Inbound-delay rule as a yes/no rule at its validation threshold (>= 5 min): on April it flags 25.4% of
  flights with precision 0.340 and recall 0.452 (F1 0.388).
- LightGBM confusion matrix at 20%: TN 414,914, FP 52,053, FN 46,618, TP 63,330.
- **LightGBM vs logistic regression** (paired bootstrap over the 30 April days, 200 resamples):
  AUC-ROC +0.0446 (95% CI +0.0422 to +0.0468), AUC-PR +0.1226 (95% CI +0.1138 to +0.1332). Unlike the
  weeks-ahead question, where the two were level, the trees now clearly win: the same-day signals
  interact (e.g. a 30-minute inbound delay matters only when the planned turn is short).
- **Gain over the obvious rule**: at the same 20% flag rate LightGBM's precision is 17.2 points and
  its recall 18.1 points higher than the inbound-delay rule's; at 10%, 29.1 and 15.3 points.

### Probabilities and calibration (April, never fitted on)

| April | Actual rate | Mean predicted | Brier score | Brier of a constant 18.54% |
|---|---|---|---|---|
| Logistic regression | 0.1906 | 0.2008 | 0.12383 | 0.15429 |
| LightGBM | 0.1906 | 0.1972 | 0.10770 | 0.15429 |

LightGBM's top decile: 77.6% predicted vs 76.5% actual. The probabilities are published as is
(unweighted log-loss, not recalibrated); the weeks-ahead version's over-spread top decile (45.7% vs
35.2%) is gone.

### What drives the prediction (SHAP)

Exact TreeSHAP (LightGBM `pred_contrib`, log-odds) on a fixed 20,000-flight April sample. Share of
total mean |SHAP| by group: schedule + history 34.3%, inbound aircraft 28.4%, congestion 18.2%,
origin weather 9.6%, destination weather (forecast stand-in) 9.5%. Top features:

| Feature | Mean \|SHAP\| |
|---|---|
| projected turn slack (min) | 0.352 |
| inbound leg departure delay (if departed by cutoff) | 0.175 |
| destination avg arrival delay, 2 h before cutoff | 0.140 |
| carrier+route delay-rate lift (prev. month) | 0.138 |
| scheduled turn after inbound leg (min) | 0.104 |
| destination arrivals scheduled that hour | 0.103 |
| distance | 0.103 |
| aircraft's latest known departure delay | 0.100 |
| origin temperature at cutoff (C) | 0.098 |
| destination precipitation at sched. arrival (mm/h) | 0.097 |

Example (highest-risk April flight): AS FLL-SAN on 2024-04-11, 17:00 departure, probability 0.999,
actually delayed; reasons: projected turn slack -49 min (+3.62), aircraft's latest known departure delay
118 min (+0.81), inbound leg departure delay 118 min (+0.69).

## Dashboards

Both builders take titles, the test month and the month order from the data (nothing hard-coded to
"April"); the visual design is unchanged.

**Excel** (`outputs/Flight_Delays_Dashboard.xlsx`, built with openpyxl): a Dashboard sheet with eight
KPI cards, all live formulas over the data sheets: total scheduled flights 2,240,444, on-time arrival
80.0%, cancellation rate 1.45%, LightGBM AUC-PR 0.633, precision 54.9% and recall 57.6% when flagging
the riskiest 20%, and precision 76.5% / recall 40.1% at the riskiest 10%. The baseline line under the
cards gives the AUC-PR of the carrier+route rule, the inbound-delay rule and logistic regression. There
are six charts (on-time % by carrier, delay rate by scheduled hour, daily on-time %, delay minutes by
cause and month, model comparison for the four models on April 2 hours before departure, most-delayed of
the 30 busiest airports). There is also a Summary sheet, one data sheet per exported table (including
`BeforeAfter`: weeks-ahead vs 2-hours-before LightGBM on the same test month), and a FlaggedFlights sheet
with the 500 highest-probability of the 115,383 flagged April flights, their inbound known delay and
their SHAP reasons.

**Tableau** (`tableau/Flight_Delays.twbx`, extract-based so it opens in Tableau Public): one dark-navy
dashboard, *US Flight Delays and Delay Model (Jan-Apr 2024)*, with the same theme as the Retail
Analytics workbook (automatic sizing, borderless tiles, no gridlines/zero lines/axis lines/dividers).
- **8 KPI tiles**: 2.24M flights, 80.0% on time, 5.7 min average arrival delay, 1.45% cancelled, least
  on-time carrier Frontier, LightGBM AUC-PR 0.633, precision 54.89% and recall 57.60% at the riskiest 20%
  (from the LightGBM confusion matrix: 63,330 / 115,383 and 63,330 / 109,948).
- **6 charts (3 x 2)**: delay rate by day of week and month (heatmap), on-time % by carrier, weekly
  delay rate, 20 busiest airports shaded by delay rate (treemap); *Apr 2024: Predicted vs Actual by
  Carrier* (scatter of the 2-hours-before LightGBM's mean probability vs the actual rate per carrier); and
  *Apr 2024, 2 h Before: Top-20% Flags* (AUC-PR, precision and recall for a random ranking, the
  carrier+route rule, the inbound-delay rule, logistic regression and LightGBM).
- Known cosmetic limit: in the scatter, the names of the clustered mid-range carriers overlap; hover
  shows each carrier.

## Limitations (honest)

- **Different question, not a better answer to the old one.** Two hours before departure the
  inbound aircraft and the day's operations are visible, which is why the scores roughly double. For
  planning weeks ahead the old numbers (AUC-ROC about 0.65) still apply.
- **Tail-number identity is after-the-fact.** BTS records the aircraft that actually flew; delay-driven
  swaps show up as impossible turns. Excluding those 1.8% of flights lowers LightGBM from 0.809 / 0.633
  to 0.796 / 0.594 AUC-ROC / AUC-PR (see "Residual leakage risk"), so the headline is slightly optimistic.
- **Destination weather is observed, not forecast.** Historical weather at the scheduled arrival hour
  stands in for the forecast an airline would have at the cutoff. On validation this group adds only
  +0.006 AUC-ROC / +0.005 AUC-PR, so the optimism is small, but it is real. Origin weather uses only
  hours up to the cutoff. Open-Meteo's archive is reanalysis (ERA5-based) gridded weather, not the
  airport's METAR, and has no visibility; low cloud cover is the ceiling proxy.
- **Event times are reconstructed.** Actual departure/arrival = scheduled time + BTS delay minutes (gate
  out / gate in). Real-time feeds would carry a few minutes of reporting lag; the rules use strict
  "before the cutoff" but do not add a lag margin.
- **Offline operating point.** Flagging the riskiest 20% / 10% of a month is a comparison convention;
  a live system would use a probability threshold, and flights 2 hours out arrive continuously.
- **Only four months.** One history, one fit, one validation and one test month; base rates swing
  16-24% between months, so every number carries month-to-month noise.
- **Not tuned**: LightGBM hyperparameters are the weeks-ahead defaults, not searched for the new
  features.
- **Overnight / date-line arrivals**: scheduled UTC arrival = departure + block time (robust); the local
  arrival date used for destination-hour volumes still relies on the +/-12 h inference, which is wrong for
  the few date-line flights (HNL-GUM), a negligible effect on one volume feature.

## Verified run evidence

Run on 2026-09-26 (all pipeline commands inside the project's Docker stack):

- **Full DAG**: `airflow dags trigger flight_delay_pipeline_dag`, run `manual__2026-09-26T11:31:35+00:00`,
  state `success`. `airflow tasks states-for-dag-run`: all 12 tasks `success` (wait_for_raw_dir,
  ingest_raw_flights, fetch_weather, great_expectations_validate, dbt_run, build_cutoff_features,
  leakage_check, dbt_test, train_delay_model, export_dashboard_data, build_excel_dashboard,
  build_tableau_workbook), 11:31:36 to 11:51:47 UTC (about 20 minutes; train_delay_model 13 minutes, of which
  about 5 are the per-flight SHAP reasons for the 115,383 flagged flights). ingest reloaded 2,240,444 flights;
  fetch_weather found all 4 months cached and made no API calls.
- **Weather download** (first run only): 35 Open-Meteo requests (5 time-zone batches + 7 batches x 4 months),
  988,920 hourly rows for 335 airports, cached in `data/raw/weather/`.
- **Great Expectations**: PASSED (all hard gates, as before).
- **Leakage check**: `LEAKAGE CHECK PASSED` in the DAG (0 timestamp violations over 2,202,967 rows, 0 key
  cross-check violations, 0 mismatches in 8,400 independently recomputed values); report in
  `outputs/leakage_check.json`.
- **dbt test**: `Done. PASS=66 WARN=0 ERROR=0 SKIP=0 TOTAL=66` (52 previous tests + UTC columns, new sources,
  and the 3 leakage / time-zone singular tests).
- **Reproducibility**: `train_delay_model.py` was run twice by hand and a third time by the DAG;
  `model_metrics.json` was identical all three times (full JSON equality) and
  `predicted_delays_test_month.csv` was byte-identical.
- **Design checks**: `design_checks.py` ran on Feb (fit) / Mar (validation) only; the ablation table above is
  copied from `outputs/design_checks.json`.
- **Excel**: opened read-only in Excel and recalculated (scratchpad test script): KPIs 2,240,444 / 80.0% /
  1.45% / 0.633 / 54.9% / 57.6% / 76.5% / 40.1%, 0 formula errors, 6 charts, all exported and checked by eye
  (charts 5-6 are below the first screen and exported blank from the test script, so they were exported
  again after scrolling; the model chart shows exactly the April values in the results table).
- **Tableau**: the DAG built the workbook; the two bottom-right chart titles were then shortened to fit one
  line and the workbook rebuilt on the host with the same script (`build_tableau_workbook.py`, it only
  reads `data/processed/`). Opened in Tableau Public: load check PASS (workbook-load-completed logged, no
  error dialog; the only error-level log lines were OpenGL rendering messages). The full-window
  screenshot was checked by eye: navy theme unchanged, eight untruncated KPIs (2.24M, 80.0%, 5.7 min,
  1.45%, Frontier, 0.633, 54.89%, 57.60%), readable single-line titles, and all six charts showing the
  `dash_*.csv` values, including the inbound rule in the model chart.
