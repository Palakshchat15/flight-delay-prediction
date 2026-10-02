{{ config(materialized='table') }}
-- Clean types and dedupe raw.flights. One row per scheduled flight.
-- Natural key: date + carrier + flight number + origin + scheduled departure.
-- BTS writes end-of-day midnight as '2400'; it is mapped to '2359' so it stays on the same date/hour.
with src as (
    select
        f.*,
        case when f.crs_dep_time = '2400' then '2359' else f.crs_dep_time end as crs_dep_hhmm,
        case when f.crs_arr_time = '2400' then '2359' else f.crs_arr_time end as crs_arr_hhmm
    from {{ source('raw', 'flights') }} f
),
ranked as (
    select
        *,
        row_number() over (
            partition by flight_date, reporting_airline, flight_number, origin, crs_dep_time
            order by loaded_at desc, source_file desc
        ) as rn
    from src
)
select
    md5(concat_ws('|', flight_date, reporting_airline, flight_number, origin, crs_dep_time)) as flight_key,
    flight_date,
    year::int                                   as flight_year,
    month::int                                  as flight_month,
    day_of_month::int                           as day_of_month,
    day_of_week::int                            as day_of_week,      -- 1 = Monday ... 7 = Sunday (BTS convention)
    reporting_airline                           as carrier_code,
    nullif(tail_number, '')                     as tail_number,
    flight_number,
    origin,
    origin_city_name,
    origin_state,
    dest,
    dest_city_name,
    dest_state,
    origin || '-' || dest                       as route,
    crs_dep_hhmm                                as crs_dep_time,
    (substr(crs_dep_hhmm, 1, 2))::int           as dep_hour,
    (substr(crs_dep_hhmm, 1, 2))::int * 60 + (substr(crs_dep_hhmm, 3, 2))::int as dep_minute_of_day,
    crs_arr_hhmm                                as crs_arr_time,
    (substr(crs_arr_hhmm, 1, 2))::int           as arr_hour,
    crs_elapsed_time::int                       as crs_elapsed_minutes,
    distance::int                               as distance_miles,
    (cancelled = 1)                             as is_cancelled,
    (diverted = 1)                              as is_diverted,
    nullif(cancellation_code, '')               as cancellation_code,
    case cancellation_code
        when 'A' then 'Carrier' when 'B' then 'Weather'
        when 'C' then 'National Air System' when 'D' then 'Security'
    end                                         as cancellation_reason,
    dep_delay::int                              as dep_delay_minutes,
    arr_delay::int                              as arr_delay_minutes,
    arr_del15::int                              as is_delayed_15,
    coalesce(carrier_delay, 0)::int             as carrier_delay_minutes,
    coalesce(weather_delay, 0)::int             as weather_delay_minutes,
    coalesce(nas_delay, 0)::int                 as nas_delay_minutes,
    coalesce(security_delay, 0)::int            as security_delay_minutes,
    coalesce(late_aircraft_delay, 0)::int       as late_aircraft_delay_minutes,
    source_file
from ranked
where rn = 1
