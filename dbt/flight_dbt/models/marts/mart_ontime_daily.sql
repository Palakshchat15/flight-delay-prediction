select
    flight_date,
    {{ delay_metrics() }}
from {{ ref('fact_flights') }}
group by 1
