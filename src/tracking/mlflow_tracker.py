# src/tracking/mlflow_tracker.py

import joblib
import mlflow
import mlflow.sklearn

model = joblib.load(
    "models/best_model.pkl"
)

with mlflow.start_run():

    mlflow.log_param(
        "model_name",
        "Best Model"
    )

    mlflow.sklearn.log_model(
        model,
        "forecast_model"
    )

print("MLflow tracking completed")