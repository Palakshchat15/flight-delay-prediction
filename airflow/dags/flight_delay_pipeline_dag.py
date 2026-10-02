"""
Airline Flight Delay pipeline DAG.

wait_for_raw_files (PythonSensor) -> ingest_raw_flights (BTS zips -> raw.flights,
OurAirports -> raw.airports) -> fetch_weather (Open-Meteo, cached) ->
great_expectations_validate -> dbt_run (seed + run) -> build_cutoff_features
(features known 2 h before departure) -> leakage_check -> dbt_test ->
train_delay_model -> export_dashboard_data -> [build_excel_dashboard, build_tableau_workbook]

The sensor only waits for the raw directory to be mounted; missing monthly zips
are downloaded by the ingestion task itself.
"""
import logging
import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor

logger = logging.getLogger(__name__)
PROJECT_ROOT = "/opt/airflow"
DBT_DIR = f"{PROJECT_ROOT}/dbt/flight_dbt"
DBT_ENV = f"DBT_PROFILES_DIR={PROJECT_ROOT}/dbt"

default_args = {
    "owner": "flight-delay-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def on_failure_callback(context):
    ti = context.get("task_instance")
    logger.error("Task failed: dag=%s task=%s run=%s try=%s",
                 context["dag"].dag_id, ti.task_id if ti else "?",
                 context.get("run_id"), ti.try_number if ti else "?")


def raw_dir_ready():
    return os.path.isdir(f"{PROJECT_ROOT}/data/raw")


def run_ge_validation():
    result = subprocess.run(["python", f"{PROJECT_ROOT}/great_expectations/validate_raw_flights.py"],
                            capture_output=True, text=True, timeout=1800)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError("Great Expectations validation failed")


with DAG(
    dag_id="flight_delay_pipeline_dag",
    default_args=default_args,
    description="BTS on-time data + weather -> GE -> dbt -> 2h-before-departure delay model -> Excel + Tableau",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["flight-delay", "portfolio"],
    on_failure_callback=on_failure_callback,
) as dag:

    wait_for_raw = PythonSensor(
        task_id="wait_for_raw_dir",
        python_callable=raw_dir_ready,
        poke_interval=30,
        timeout=600,
        mode="reschedule",
    )

    ingest = BashOperator(
        task_id="ingest_raw_flights",
        bash_command=f"python {PROJECT_ROOT}/src/ingestion/load_raw_flights.py",
        execution_timeout=timedelta(minutes=60),
    )

    fetch_weather = BashOperator(
        task_id="fetch_weather",
        bash_command=f"python {PROJECT_ROOT}/src/ingestion/fetch_weather.py",
        execution_timeout=timedelta(minutes=60),
    )

    ge_validate = PythonOperator(
        task_id="great_expectations_validate",
        python_callable=run_ge_validation,
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {DBT_DIR} && {DBT_ENV} dbt seed && {DBT_ENV} dbt run",
    )

    cutoff_features = BashOperator(
        task_id="build_cutoff_features",
        bash_command=f"cd {PROJECT_ROOT}/src/ml_pipeline && python cutoff_features.py",
        execution_timeout=timedelta(minutes=30),
    )

    leakage_check = BashOperator(
        task_id="leakage_check",
        bash_command=f"cd {PROJECT_ROOT}/src/ml_pipeline && python leakage_check.py",
        execution_timeout=timedelta(minutes=30),
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {DBT_DIR} && {DBT_ENV} dbt test",
    )

    train_model = BashOperator(
        task_id="train_delay_model",
        bash_command=f"cd {PROJECT_ROOT}/src/ml_pipeline && python train_delay_model.py",
        execution_timeout=timedelta(minutes=60),
    )

    export_data = BashOperator(
        task_id="export_dashboard_data",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python export_dashboard_data.py",
    )

    build_excel = BashOperator(
        task_id="build_excel_dashboard",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python build_excel_dashboard.py",
    )

    build_tableau = BashOperator(
        task_id="build_tableau_workbook",
        bash_command=f"cd {PROJECT_ROOT}/src/dashboards && python build_tableau_workbook.py",
    )

    wait_for_raw >> ingest >> fetch_weather >> ge_validate >> dbt_run >> cutoff_features >> leakage_check
    leakage_check >> dbt_test >> train_model >> export_data
    export_data >> [build_excel, build_tableau]
