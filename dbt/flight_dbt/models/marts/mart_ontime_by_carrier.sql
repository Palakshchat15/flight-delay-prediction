select
    c.carrier_code,
    c.carrier_name,
    {{ delay_metrics() }}
from {{ ref('fact_flights') }} f
join {{ ref('dim_carrier') }} c using (carrier_code)
group by 1, 2
