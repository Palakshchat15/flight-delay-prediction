-- Leakage gate: every event a 2-hours-before-departure feature used must have happened
-- strictly before the flight's cutoff (scheduled departure - 2 h). Returns violating rows.
select flight_key, cutoff_utc
from {{ source('features', 'flight_cutoff_features') }}
where inb_prev_dep_ts       >= cutoff_utc
   or inb_prev_arr_ts       >= cutoff_utc
   or ac_latest_dep_ts      >= cutoff_utc
   or ac_latest_arr_ts      >= cutoff_utc
   or origin_dep_win_max_ts >= cutoff_utc
   or origin_arr_win_max_ts >= cutoff_utc
   or dest_arr_win_max_ts   >= cutoff_utc
   or dest_dep_win_max_ts   >= cutoff_utc
   or carrier_win_max_ts    >= cutoff_utc
   or nat_win_max_ts        >= cutoff_utc
   or wx_o_ts               >  cutoff_utc
