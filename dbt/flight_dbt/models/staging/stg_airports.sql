-- One row per IATA code. A handful of codes appear more than once in
-- OurAirports (closed fields, heliports); prefer US, then larger airport types.
with ranked as (
    select
        iata_code,
        ident,
        name,
        municipality,
        iso_country,
        replace(iso_region, 'US-', '')  as state_code,
        airport_type,
        latitude_deg,
        longitude_deg,
        row_number() over (
            partition by iata_code
            order by (iso_country = 'US') desc,
                     case airport_type when 'large_airport' then 1 when 'medium_airport' then 2
                                       when 'small_airport' then 3 else 4 end,
                     ident
        ) as rn
    from {{ source('raw', 'airports') }}
)
select iata_code, ident, name, municipality, iso_country, state_code, airport_type, latitude_deg, longitude_deg
from ranked
where rn = 1
