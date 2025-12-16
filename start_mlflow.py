#!/usr/bin/env python3
"""
Simple MLflow UI launcher that avoids PySpark issues
"""
import sys
import os

# Disable PySpark to avoid compatibility issues
os.environ['DISABLE_MLFLOW_INTEGRATION'] = '1'
os.environ['MLFLOW_DISABLE_ENV_MANAGER_CONDA_DEPENDENCIES'] = 'TRUE'

# Remove pyspark from sys.modules if it tries to load
sys.modules['pyspark'] = None

# Now import MLflow
import mlflow.server

if __name__ == '__main__':
    # Start MLflow UI
    mlflow.server._run_server(
        backend_store_uri='./mlruns',
        default_artifact_root=None,
        serve_artifacts=False,
        artifacts_only=False,
        host='127.0.0.1',
        port=5000,
        workers=1,
        static_prefix=None,
        gunicorn_opts=None,
        waitress_opts=None,
        expose_prometheus=None,
        artifacts_destination=None,
        app_name='mlflow'
    )
