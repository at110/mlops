import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

os.environ[
    "MLFLOW_TRACKING_URI"
] = "sqlite:////Users/ashishtamhane/mlflow_postgres_minio/mlflow/mlflow.db"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

experiment_name = "demo_experiment7"
try:
    mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow/")
except MlflowException as e:
    print(e)
mlflow.set_experiment(experiment_name)

mlflow.start_run()

# Log a parameter (key-value pair)
mlflow.log_param("param1", 5)
# Log a metric; metrics can be updated throughout the run
mlflow.log_metric("foo", 1)
mlflow.log_metric("foo", 2)
mlflow.log_metric("foo", 3)
# Log an artifact (output file)
with open("output.txt", "w") as f:
    f.write("Hello world!")
mlflow.log_artifact("output.txt")

mlflow.end_run()
