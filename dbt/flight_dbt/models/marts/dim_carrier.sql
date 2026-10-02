select
    f.carrier_code,
    coalesce(c.carrier_name, f.carrier_code)  as carrier_name,
    count(*)                                  as scheduled_flights
from {{ ref('stg_flights') }} f
left join {{ ref('carriers') }} c on c.carrier_code = f.carrier_code
group by 1, 2
