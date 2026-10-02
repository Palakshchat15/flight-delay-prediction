FROM apache/airflow:2.9.3-python3.11

USER root
# libgomp1 is the OpenMP runtime LightGBM links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt
