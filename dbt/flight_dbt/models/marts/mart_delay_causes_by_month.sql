-- BTS attributes the arrival-delay minutes of every 15+ min late flight to five causes.
-- Descriptive only: these are post-flight fields and are never model features.
select
    flight_month,
    to_char(make_date(2000, flight_month, 1), 'Mon') as month_name,
    cause,
    sum(minutes) as delay_minutes
from {{ ref('fact_flights') }}
cross join lateral (values
    ('Carrier', carrier_delay_minutes),
    ('Weather', weather_delay_minutes),
    ('National Air System', nas_delay_minutes),
    ('Security', security_delay_minutes),
    ('Late Aircraft', late_aircraft_delay_minutes)
) as c(cause, minutes)
where is_delayed_15 = 1
group by 1, 2, 3
