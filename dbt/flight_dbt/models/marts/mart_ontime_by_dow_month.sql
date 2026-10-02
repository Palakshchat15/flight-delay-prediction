select
    flight_month,
    to_char(make_date(2000, flight_month, 1), 'Mon')  as month_name,
    day_of_week,
    (array['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])[day_of_week] as day_name,
    {{ delay_metrics() }}
from {{ ref('fact_flights') }}
group by 1, 2, 3, 4
