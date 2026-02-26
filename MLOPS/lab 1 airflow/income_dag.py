from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from income_pipeline import run_load, run_clean, run_features, run_encode, run_train

with DAG(
        dag_id="income_prediction_pipe",
        start_date=datetime(2025, 1, 1),
        schedule="@once",
        catchup=False,
        max_active_tasks=1,
        max_active_runs=1,
        tags=["ml", "income", "classification"]
) as dag:

    load = PythonOperator(
        task_id="load_data",
        python_callable=run_load
    )

    clean = PythonOperator(
        task_id="clean_data",
        python_callable=run_clean
    )

    features = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_features
    )

    encode = PythonOperator(
        task_id="encode_scale",
        python_callable=run_encode
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=run_train
    )

    load >> clean >> features >> encode >> train