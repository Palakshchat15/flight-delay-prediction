select
    a.airport_code,
    a.airport_name,
    a.city_name,
    a.state_code,
    a.latitude,
    a.longitude,
    {{ delay_metrics() }}
from {{ ref('fact_flights') }} f
join {{ ref('dim_airport') }} a on a.airport_code = f.origin_airport_code
group by 1, 2, 3, 4, 5, 6
