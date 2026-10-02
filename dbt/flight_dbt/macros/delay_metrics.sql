{# Shared KPI definitions for the dashboard marts (source: fact_flights).
   on_time_pct / delay_rate are over COMPLETED flights (not cancelled, not
   diverted), i.e. the flights that have an arrival time.
   cancellation_rate is over all SCHEDULED flights. #}
{% macro delay_metrics() -%}
    count(*)                                                          as scheduled_flights,
    sum(case when is_cancelled then 1 else 0 end)                     as cancelled_flights,
    sum(case when is_diverted then 1 else 0 end)                      as diverted_flights,
    sum(case when not is_cancelled and not is_diverted then 1 else 0 end) as completed_flights,
    sum(case when not is_cancelled and not is_diverted then is_delayed_15 else 0 end) as delayed_flights,
    round(avg(case when is_cancelled then 1.0 else 0.0 end), 4)       as cancellation_rate,
    round(avg(case when not is_cancelled and not is_diverted then is_delayed_15 end), 4) as delay_rate,
    round(1 - avg(case when not is_cancelled and not is_diverted then is_delayed_15 end), 4) as on_time_pct,
    round(avg(case when not is_cancelled and not is_diverted then arr_delay_minutes end), 2) as avg_arr_delay_minutes
{%- endmacro %}
