import mlflow

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Deep Learning Experiments"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("MLflow Tracking URI:", mlflow.get_tracking_uri())
print("Experiment:", EXPERIMENT_NAME)
