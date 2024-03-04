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

"""
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
"""
def load_and_split_data(csv_url: str, test_size: float = 0.25, random_state: int = 42) -> tuple:
    """
    Load data from CSV URL and split it into training and testing datasets.

    Parameters:
    - csv_url (str): URL to the CSV file.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    - tuple: (train_x, train_y, test_x, test_y) datasets.
    """
    data = pd.read_csv(csv_url, sep=";")
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    train_x, train_y = train.drop(["quality"], axis=1), train[["quality"]]
    test_x, test_y = test.drop(["quality"], axis=1), test[["quality"]]
    return train_x, train_y, test_x, test_y

def train_model(train_x: pd.DataFrame, train_y: pd.DataFrame, alpha: float, l1_ratio: float) -> ElasticNet:
    """
    Train an ElasticNet model.

    Parameters:
    - train_x (pd.DataFrame): Features for training.
    - train_y (pd.DataFrame): Target variable for training.
    - alpha (float): Constant that multiplies the penalty terms.
    - l1_ratio (float): The ElasticNet mixing parameter.

    Returns:
    - ElasticNet: The trained model.
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(train_x, train_y)
    return model

def evaluate_model(model, test_x: pd.DataFrame, test_y: pd.DataFrame) -> tuple:
    """
    Evaluate the model on the test dataset.

    Parameters:
    - model: The trained model.
    - test_x (pd.DataFrame): Features for testing.
    - test_y (pd.DataFrame): True values for testing.

    Returns:
    - tuple: Evaluation metrics (RMSE, MAE, R2).
    """
    predictions = model.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    return rmse, mae, r2

def log_results(model, alpha: float, l1_ratio: float, rmse: float, mae: float, r2: float):
    """
    Log parameters, metrics, and model to MLflow.

    Parameters:
    - model: The trained model.
    - alpha (float): Constant that multiplies the penalty terms.
    - l1_ratio (float): The ElasticNet mixing parameter.
    - rmse (float): Root Mean Squared Error.
    - mae (float): Mean Absolute Error.
    - r2 (float): R2 score.
    """
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.sklearn.log_model(model, "model")

def train_and_log(in_alpha: float, in_l1_ratio: float):
    """
    Main function to handle the workflow of loading data, training the model,
    evaluating it, and logging the results.

    Parameters:
    - in_alpha (float): The alpha value for the ElasticNet model.
    - in_l1_ratio (float): The l1_ratio value for the ElasticNet model.
    """
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    train_x, train_y, test_x, test_y = load_and_split_data(csv_url)
    model = train_model(train_x, train_y, in_alpha, in_l1_ratio)
    rmse, mae, r2 = evaluate_model(model, test_x, test_y)
    log_results(model, in_alpha, in_l1_ratio, rmse, mae, r2)

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
    run_name = f"alpha_{alpha}_l1ratio_{l1_ratio}"
    with mlflow.start_run(run_name=run_name):
        train_and_log(alpha, l1_ratio)
