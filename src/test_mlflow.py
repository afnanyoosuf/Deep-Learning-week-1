import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("Deep Learning Experiments")

with mlflow.start_run(run_name="first-test-run"):

    mlflow.log_param("framework", "PyTorch")
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)

    mlflow.log_metric("train_loss", 0.45)
    mlflow.log_metric("validation_loss", 0.52)
    mlflow.log_metric("accuracy", 0.87)

    mlflow.set_tag("stage", "development")
    mlflow.set_tag("model_type", "neural_network")

print("MLflow run completed successfully.")
