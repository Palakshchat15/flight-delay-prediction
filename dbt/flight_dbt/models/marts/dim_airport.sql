-- Every airport that appears as an origin or destination, with OurAirports coordinates.
-- Matched on IATA code, falling back to the US ICAO ident ('K' + code): OurAirports
-- relabels codes after renamings (e.g. PBI's row now carries IATA 'DJT', ident 'KPBI').
with codes as (
    select origin as airport_code, origin_city_name as city_name, origin_state as state_code from {{ ref('stg_flights') }}
    union
    select dest, dest_city_name, dest_state from {{ ref('stg_flights') }}
),
one_row as (
    select airport_code, min(city_name) as city_name, min(state_code) as state_code
    from codes group by airport_code
)
select
    o.airport_code,
    coalesce(a.name, k.name, o.airport_code)       as airport_name,
    o.city_name,
    o.state_code,
    coalesce(a.latitude_deg, k.latitude_deg)       as latitude,
    coalesce(a.longitude_deg, k.longitude_deg)     as longitude
from one_row o
left join {{ ref('stg_airports') }} a on a.iata_code = o.airport_code
left join {{ ref('stg_airports') }} k on k.ident = 'K' || o.airport_code and a.iata_code is null
