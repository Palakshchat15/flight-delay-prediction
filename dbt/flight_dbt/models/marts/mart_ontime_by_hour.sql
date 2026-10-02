-- By scheduled (CRS) departure hour, local time at the origin.
select
    dep_hour,
    {{ delay_metrics() }}
from {{ ref('fact_flights') }}
group by 1
