import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import unittest
import env_variables


def eval_metrics(actual: np.ndarray, pred: np.ndarray) -> tuple:
    """
    Evaluate regression metrics.

    Parameters:
    actual (np.ndarray): Actual values.
    pred (np.ndarray): Predicted values.

    Returns:
    tuple: Returns the RMSE, MAE, and R2 metrics.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(in_alpha: float, in_l1_ratio: float) -> None:
    """
    Train an ElasticNet model and log metrics, parameters, and the model to MLflow.

    Parameters:
    in_alpha (float): The alpha value for the ElasticNet model.
    in_l1_ratio (float): The l1_ratio value for the ElasticNet model.
    """
    np.random.seed(40)

    # Load and split the dataset
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";")
    train, test = train_test_split(data)

    # Prepare training and testing data
    train_x, test_x = train.drop(["quality"], axis=1), test.drop(["quality"], axis=1)
    train_y, test_y = train[["quality"]], test[["quality"]]

    # Initialize and train the ElasticNet model
    lr = ElasticNet(alpha=in_alpha, l1_ratio=in_l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)

    # Evaluate the model
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    # Log metrics, parameters, and model to MLflow
    mlflow.log_param("alpha", in_alpha)
    mlflow.log_param("l1_ratio", in_l1_ratio)
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.sklearn.log_model(lr, "model")

    # Optionally, save metrics to a CSV file and log it as an artifact
    metrics_df = pd.DataFrame({"RMSE": [rmse], "MAE": [mae], "R2": [r2]})
    metrics_df.to_csv('metric.csv', index=False)
    mlflow.log_artifact("metric.csv")

if __name__ == "__main__":
    experiment_name = sys.argv[1]
    alpha = float(sys.argv[2])
    l1_ratio = float(sys.argv[3])

    # Set up MLflow experiment
    try:
        mlflow.create_experiment(experiment_name, artifact_location="s3://mlflow")
    except MlflowException as e:
        print(e)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        train(alpha, l1_ratio)
