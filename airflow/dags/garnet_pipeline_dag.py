from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

default_args = {
    "owner": "garnet",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="garnet_pipeline",
    default_args=default_args,
    description="Garnet retraining pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 4, 22),
    catchup=False,
    tags=["garnet", "training"],
) as dag:

    dvc_repro = BashOperator(
        task_id="dvc_repro",
        bash_command=(
            f"cd {PROJECT} && "
            f"MLFLOW_TRACKING_URI=http://mlflow-server:5000 "
            f"dvc repro"))

    git_commit = BashOperator(
        task_id="git_commit",
        bash_command=(
        f"cd {PROJECT} && "
        f"git add data/model_registry_log.txt dvc.lock && "
        f"git diff --cached --quiet || "
        f"(git commit -m 'ci: training pipline runned') || true"))
    
    dvc_repro >> git_commit