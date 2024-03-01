import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import os
import mlflow
import mlflow.sklearn
import sys

os.environ[
    "MLFLOW_TRACKING_URI"
] = "sqlite:////Users/ashishtamhane/mlflow_postgres_minio/mlflow/mlflow.db"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

experiment_name = sys.argv[1]
alpha = float(sys.argv[2])
l1_ratio = float(sys.argv[3])
try:
    mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow/")
except MlflowException as e:
    print(e)
mlflow.set_experiment(experiment_name)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train(in_alpha, in_l1_ratio):
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Set default values if no alpha is provided
    if float(in_alpha) is None:
        alpha = 0.5
    else:
        alpha = float(in_alpha)

    # Set default values if no l1_ratio is provided
    if float(in_l1_ratio) is None:
        l1_ratio = 0.5
    else:
        l1_ratio = float(in_l1_ratio)

    # Useful for multiple runs
    with mlflow.start_run(run_name = 'Alpha='+str(alpha)+'_l1Ration='+str(l1_ratio)):


        mlflow.log_input(train, context='training')
        mlflow.log_input(test, context='test')
        
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        mlflow.log_input(predicted_qualities, context='predictions')
        
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "model")
        df = pd.DataFrame(columns=['rmse','r2','mae'])
        df.loc[len(df.index)] = [rmse, r2, mae]
        df.to_csv('metric.csv',index=False)
        mlflow.log_artifact("metric.csv")


train(alpha, l1_ratio)
