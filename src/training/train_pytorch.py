import mlflow
import torch

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("PyTorch Experiments")

learning_rate = 0.001
batch_size = 32
epochs = 10

with mlflow.start_run(run_name="pytorch-baseline"):

    mlflow.log_params({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    })

    for epoch in range(epochs):

        # Training code
        train_loss = 0.45
        validation_loss = 0.50
        accuracy = 0.87

        mlflow.log_metrics({
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "accuracy": accuracy
        }, step=epoch)

    mlflow.set_tag("framework", "PyTorch")
