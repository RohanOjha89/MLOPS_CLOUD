from airflow import DAG
from datetime import datetime
from airflow.providers.amazon.aws.operators.ecs import EcsRunTaskOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

AWS_REGION = "us-east-1"

# --- Your AWS resources ---
ECS_CLUSTER = "mlops-cloud-cluster"
TASK_DEFINITION = "mlops-cloud-train-task"   # or "mlops-cloud-train-task:REVISION"
CONTAINER_NAME = "training"                  # must match container name in task definition

S3_BUCKET = "roh-mlops-cloud"
S3_PREFIX = "mlops-cloud"

# --- Networking (Fargate in private subnets) ---
SUBNETS = [
    "subnet-071a9e87f1459ae19",
    "subnet-0d955939c035dcbcc",
    # "subnet-PRIVATE1",
    # "subnet-PRIVATE2",
]
# SECURITY_GROUPS = ["sg-YOUR_ECS_TASK_SG"]
SECURITY_GROUPS = ["sg-0c84776bf4a0b2528"]

with DAG(
    dag_id="Airflow_ECS_Fargate_Training",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ecs", "fargate", "mlops"],
) as dag:

    run_id = "{{ ts_nodash }}"
    metrics_key = f"{S3_PREFIX}/metrics/{run_id}.json"
    model_key = f"{S3_PREFIX}/artifacts/{run_id}/kmeans.pkl"

    run_training_on_fargate = EcsRunTaskOperator(
        task_id="run_training_on_fargate",
        cluster=ECS_CLUSTER,
        task_definition=TASK_DEFINITION,
        launch_type="FARGATE",
        region_name=AWS_REGION,
        aws_conn_id="aws_default",
        network_configuration={
            "awsvpcConfiguration": {
                "subnets": SUBNETS,
                "securityGroups": SECURITY_GROUPS,
                "assignPublicIp": "DISABLED",
            }
        },
        overrides={
            "containerOverrides": [
                {
                    "name": CONTAINER_NAME,
                    "environment": [
                        {"name": "S3_BUCKET", "value": S3_BUCKET},
                        {"name": "S3_PREFIX", "value": S3_PREFIX},
                        {"name": "RUN_ID", "value": run_id},
                    ],
                }
            ]
        },
        wait_for_completion=True,
    )

    wait_for_metrics = S3KeySensor(
        task_id="wait_for_metrics",
        bucket_name=S3_BUCKET,
        bucket_key=metrics_key,
        aws_conn_id="aws_default",
        timeout=60 * 20,
        poke_interval=20,
    )

    wait_for_model = S3KeySensor(
        task_id="wait_for_model",
        bucket_name=S3_BUCKET,
        bucket_key=model_key,
        aws_conn_id="aws_default",
        timeout=60 * 20,
        poke_interval=20,
    )

    run_training_on_fargate >> [wait_for_metrics, wait_for_model]
