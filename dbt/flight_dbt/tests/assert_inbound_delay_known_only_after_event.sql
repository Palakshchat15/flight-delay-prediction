-- Leakage gate: the inbound leg's departure (arrival) delay is filled only when that leg
-- actually departed (arrived) before this flight's cutoff, and equals the recorded value.
select f.flight_key
from {{ source('features', 'flight_cutoff_features') }} f
left join {{ ref('int_flights_enriched') }} p on p.flight_key = f.prev_flight_key
where (f.inb_prev_dep_delay is not null) <> coalesce(p.actual_dep_utc < f.cutoff_utc, false)
   or (f.inb_prev_arr_delay is not null) <> coalesce(p.actual_arr_utc < f.cutoff_utc, false)
   or f.inb_prev_dep_delay <> p.dep_delay_minutes
   or f.inb_prev_arr_delay <> p.arr_delay_minutes
   or f.prev_flight_key = f.flight_key
