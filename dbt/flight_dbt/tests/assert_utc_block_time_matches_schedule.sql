-- Time-zone sanity: the block time implied by the local clock times + airport zones must match BTS's scheduled elapsed time
-- (within 5 min) for at least 99% of flights; otherwise airport time zones are wrong.
select share_ok
from (
    select avg((abs(extract(epoch from (sched_arr_utc_local_clock - sched_dep_utc)) / 60 - crs_elapsed_minutes) <= 5)::int) as share_ok
    from {{ ref('int_flights_enriched') }}
    where crs_elapsed_minutes is not null
) t
where share_ok < 0.99
