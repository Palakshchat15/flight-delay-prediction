{# Use the custom schema exactly as specified in each model's config
   (staging/intermediate/warehouse), rather than dbt's default behavior of
   concatenating it with the profile's target schema. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
