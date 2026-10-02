-- Routes with at least 500 completed flights in the period, ranked by delay rate.
with routes as (
    select
        route,
        origin_airport_code,
        dest_airport_code,
        {{ delay_metrics() }}
    from {{ ref('fact_flights') }}
    group by 1, 2, 3
)
select
    row_number() over (order by delay_rate desc, completed_flights desc, route) as delay_rank,
    *
from routes
where completed_flights >= 500
