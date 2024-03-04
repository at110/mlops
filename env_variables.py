# Configure environment variables for MLflow
import os
os.environ["MLFLOW_TRACKING_URI"] = "postgresql://user:password@localhost:5435/mlflowdb"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9001"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
