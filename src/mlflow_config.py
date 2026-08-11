import os
import mlflow


# Use the environment variable when running inside Docker.
# Fall back to localhost for normal local development.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

EXPERIMENT_NAME = "Deep Learning Experiments"


# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", EXPERIMENT_NAME)
