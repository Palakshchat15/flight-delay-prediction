{{ config(materialized='table') }}
-- Adds flight status plus schedule-derived context. Every feature here is
-- computed from the published schedule over ALL scheduled flights (including
-- ones later cancelled), so it is known before departure:
--   * aircraft_leg_of_day: the tail number's nth scheduled flight that day
--   * sched_turnaround_minutes: gap between the aircraft's previous scheduled
--     arrival and this scheduled departure (same airport, same local clock)
--   * origin/dest scheduled volume in the same hour (congestion proxy)
-- Destination volume is counted on the scheduled ARRIVAL date: ~3% of flights
-- are scheduled to arrive after local midnight, so their arrival hour belongs to
-- the next day. BTS gives local hhmm times but no arrival date, so the day offset
-- is recovered from departure time + scheduled block time vs the local arrival
-- clock: offset = round((dep_min + elapsed - arr_min) / 1440). Time-zone
-- differences (< 12 h within the US mainland/AK/HI) do not flip the rounding.
--
-- UTC timeline (for the "2 hours before scheduled departure" features): BTS times are
-- local clock times, so each scheduled time is converted with its airport's IANA time
-- zone (raw.airport_timezones, from Open-Meteo), DST-aware via Postgres AT TIME ZONE:
--   sched_dep_utc  = flight_date + crs_dep_time at the origin's zone
--   sched_arr_utc  = sched_dep_utc + scheduled block time (crs_elapsed_minutes)
--   sched_arr_utc_local_clock = sched_arr_date + crs_arr_time at the destination's zone
--                    (time-zone sanity check only; wrong across the date line)
--   actual_dep_utc = sched_dep_utc + dep_delay minutes   (POST-EVENT: known only once departed)
--   actual_arr_utc = sched_arr_utc + arr_delay minutes   (POST-EVENT: known only once arrived)
--   cutoff_utc     = sched_dep_utc - 2 hours (the prediction moment)
-- The actual_* columns are never model features for the flight itself; the cutoff
-- feature builder may only use OTHER flights' actual events that happened before this
-- flight's cutoff_utc (enforced by src/ml_pipeline/leakage_check.py and dbt tests).
with base as (
    select
        s.*,
        -- (1 flight has no scheduled block time: assumed same-day arrival)
        s.flight_date + coalesce((round((s.dep_minute_of_day + s.crs_elapsed_minutes
                                - ((substr(s.crs_arr_time, 1, 2))::int * 60
                                   + (substr(s.crs_arr_time, 3, 2))::int)) / 1440.0))::int, 0)
                                                         as sched_arr_date
    from {{ ref('stg_flights') }} s
),
utc0 as (
    select
        b.*,
        ((b.flight_date + make_interval(mins => b.dep_minute_of_day)) at time zone tz_o.timezone)
            at time zone 'UTC'                                       as sched_dep_utc,
        ((b.sched_arr_date + make_interval(mins => (substr(b.crs_arr_time, 1, 2))::int * 60
                                                   + (substr(b.crs_arr_time, 3, 2))::int))
            at time zone tz_d.timezone) at time zone 'UTC'           as sched_arr_utc_local_clock
    from base b
    left join {{ source('raw', 'airport_timezones') }} tz_o on tz_o.airport_code = b.origin
    left join {{ source('raw', 'airport_timezones') }} tz_d on tz_d.airport_code = b.dest
),
utc as (
    -- UTC arrival = UTC departure + scheduled block time. The local-clock conversion above
    -- depends on the inferred arrival date, which is wrong for date-line flights (HNL-GUM
    -- "arrives" a day before it departs); it is kept only for the time-zone sanity test.
    select
        u.*,
        coalesce(u.sched_dep_utc + make_interval(mins => u.crs_elapsed_minutes),
                 u.sched_arr_utc_local_clock)                        as sched_arr_utc
    from utc0 u
),
aircraft as (
    select
        flight_key,
        row_number() over w                         as aircraft_leg_of_day,
        count(*) over (partition by tail_number, flight_date) as aircraft_legs_scheduled_today,
        lag(crs_arr_time) over w                    as prev_crs_arr_time
    from utc
    where tail_number is not null
    window w as (partition by tail_number, flight_date order by crs_dep_time, flight_key)
)
select
    b.*,
    case
        when b.is_cancelled then 'Cancelled'
        when b.is_diverted then 'Diverted'
        when b.is_delayed_15 = 1 then 'Delayed 15+ min'
        else 'On time'
    end                                                          as flight_status,
    b.sched_dep_utc + make_interval(mins => b.dep_delay_minutes)    as actual_dep_utc,
    b.sched_arr_utc + make_interval(mins => b.arr_delay_minutes)    as actual_arr_utc,
    b.sched_dep_utc - interval '2 hours'                             as cutoff_utc,
    a.aircraft_leg_of_day,
    a.aircraft_legs_scheduled_today,
    case
        when a.prev_crs_arr_time is null then null
        when b.dep_minute_of_day - ((substr(a.prev_crs_arr_time, 1, 2))::int * 60
                                    + (substr(a.prev_crs_arr_time, 3, 2))::int) < 0 then null
        else b.dep_minute_of_day - ((substr(a.prev_crs_arr_time, 1, 2))::int * 60
                                    + (substr(a.prev_crs_arr_time, 3, 2))::int)
    end                                                          as sched_turnaround_minutes,
    count(*) over (partition by b.origin, b.flight_date, b.dep_hour) as origin_sched_deps_in_hour,
    count(*) over (partition by b.dest, b.sched_arr_date, b.arr_hour) as dest_sched_arrs_in_hour,
    count(*) over (partition by b.origin, b.flight_date)             as origin_sched_deps_in_day
from utc b
left join aircraft a using (flight_key)
