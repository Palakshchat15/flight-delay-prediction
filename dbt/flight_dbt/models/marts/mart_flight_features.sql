-- Model feature mart. Only completed flights (not cancelled, not diverted):
-- the label ArrDel15 needs an arrival. Only PRE-DEPARTURE information is
-- exposed as features; post-flight columns (dep delay, taxi, actual times,
-- delay-cause minutes) are deliberately not selected here.
-- Historical features (delay rates, each scheduled flight's typical rotation)
-- are computed in the training script from the training months only.
-- The same-day aircraft columns (aircraft_leg_of_day, aircraft_legs_scheduled_today,
-- sched_turnaround_minutes) are kept for analysis but are NOT used directly as
-- model features: BTS tail numbers record the aircraft that actually flew, so
-- day-of aircraft swaps leak delay information (see README_PIPELINE.md).
select
    flight_key,
    flight_date,
    flight_year,
    flight_month,
    day_of_month,
    day_of_week,
    carrier_code,
    flight_number,
    origin,
    dest,
    route,
    dep_hour,
    arr_hour,
    dep_minute_of_day,
    crs_elapsed_minutes,
    distance_miles,
    aircraft_leg_of_day,
    aircraft_legs_scheduled_today,
    sched_turnaround_minutes,
    origin_sched_deps_in_hour,
    dest_sched_arrs_in_hour,
    origin_sched_deps_in_day,
    is_delayed_15
from {{ ref('int_flights_enriched') }}
where not is_cancelled and not is_diverted
