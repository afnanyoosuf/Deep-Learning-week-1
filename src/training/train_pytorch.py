import json
import os
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.models.pytorch_model import MNISTANN


# =========================================================
# 1. LOAD PARAMETERS
# =========================================================

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)


train_size = params["data"]["train_size"]
val_size = params["data"]["val_size"]
num_classes = params["data"]["num_classes"]

SEED = params["train"]["seed"]
batch_size = params["train"]["batch_size"]
epochs = params["train"]["epochs"]
learning_rate = params["train"]["learning_rate"]
optimizer_name = params["train"]["optimizer"]

input_size = params["model"]["input_size"]
hidden_size = params["model"]["hidden_size"]
dropout = params["model"]["dropout"]
batch_norm = params["model"]["batch_norm"]
output_size = params["model"]["output_size"]


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# 3. DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# 4. DIRECTORIES
# =========================================================

model_dir = Path("models/pytorch")
checkpoint_dir = model_dir / "checkpoints"
metrics_dir = Path("metrics")

model_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# 5. LOAD MNIST
# =========================================================

transform = transforms.ToTensor()

full_dataset = datasets.MNIST(
    root="data/raw",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data/raw",
    train=False,
    download=True,
    transform=transform
)


# =========================================================
# 6. TRAIN / VALIDATION SPLIT
# =========================================================

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)


# =========================================================
# 7. DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# =========================================================
# 8. MODEL
# =========================================================

model = MNISTANN(
    hidden_size=hidden_size,
    dropout=dropout,
    batch_norm=batch_norm
).to(device)

print("\nModel:")
print(model)


# =========================================================
# 9. LOSS
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# 10. OPTIMIZER
# =========================================================

if optimizer_name.lower() == "adam":

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

elif optimizer_name.lower() == "sgd":

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

elif optimizer_name.lower() == "rmsprop":

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=learning_rate
    )

else:

    raise ValueError(
        f"Unknown optimizer: {optimizer_name}"
    )


# =========================================================
# 11. HISTORY
# =========================================================

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}


# =========================================================
# 12. CHECKPOINT SETTINGS
# =========================================================

best_val_accuracy = 0.0

best_checkpoint_path = (
    model_dir / "best_checkpoint.pth"
)


# =========================================================
# 13. MLFLOW CONFIGURATION
# =========================================================

# Docker:
#   MLFLOW_TRACKING_URI=http://host.docker.internal:5000
#
# Local Windows:
#   Defaults to http://localhost:5000

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

MLFLOW_EXPERIMENT_NAME = (
    "Deep Learning Experiments"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)

print(
    "MLflow Tracking URI:",
    mlflow.get_tracking_uri()
)

print(
    "MLflow Experiment:",
    MLFLOW_EXPERIMENT_NAME
)


# =========================================================
# 14. START MLFLOW RUN
# =========================================================

with mlflow.start_run(
    run_name="pytorch-mnist-ann"
):

    # -----------------------------------------------------
    # MLFLOW PARAMETERS
    # -----------------------------------------------------

    mlflow.log_params({

        "seed": SEED,
        "train_size": train_size,
        "val_size": val_size,
        "num_classes": num_classes,

        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,

        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "dropout": dropout,
        "batch_norm": batch_norm,

        "device": str(device)
    })

    print("\nStarting training...")

    # =====================================================
    # 15. TRAINING LOOP
    # =====================================================

    for epoch in range(epochs):

        # -------------------------------------------------
        # TRAINING
        # -------------------------------------------------

        model.train()

        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        model.eval()

        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                running_val_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        # -------------------------------------------------
        # CALCULATE METRICS
        # -------------------------------------------------

        train_loss = (
            running_train_loss /
            len(train_loader)
        )

        val_loss = (
            running_val_loss /
            len(val_loader)
        )

        train_accuracy = (
            train_correct /
            train_total
        )

        val_accuracy = (
            val_correct /
            val_total
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        # -------------------------------------------------
        # PRINT METRICS
        # -------------------------------------------------

        print(
            f"\nEpoch [{epoch + 1}/{epochs}]"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # -------------------------------------------------
        # MLFLOW EPOCH METRICS
        # -------------------------------------------------

        mlflow.log_metrics(

            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy
            },

            step=epoch + 1
        )

        # -------------------------------------------------
        # CHECKPOINT
        # -------------------------------------------------

        checkpoint = {

            "epoch": epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "train_accuracy":
                train_accuracy,

            "val_accuracy":
                val_accuracy,

            "seed":
                SEED,

            "hidden_size":
                hidden_size,

            "dropout":
                dropout,

            "batch_norm":
                batch_norm,

            "optimizer":
                optimizer_name,

            "learning_rate":
                learning_rate,

            "batch_size":
                batch_size
        }

        epoch_checkpoint_path = (
            checkpoint_dir /
            f"checkpoint_epoch_{epoch + 1}.pth"
        )

        torch.save(
            checkpoint,
            epoch_checkpoint_path
        )

        print(
            "Checkpoint saved:",
            epoch_checkpoint_path
        )

        # -------------------------------------------------
        # BEST CHECKPOINT
        # -------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                checkpoint,
                best_checkpoint_path
            )

            print(
                "Best checkpoint saved "
                f"at epoch {epoch + 1}"
            )

    # =====================================================
    # 16. FINAL MODEL
    # =====================================================

    final_model_path = (
        model_dir /
        "mnist_ann_bn_dropout.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )

    # =====================================================
    # 17. TRAIN METRICS
    # =====================================================

    train_metrics = {

        "best_val_accuracy":
            best_val_accuracy,

        "final_train_loss":
            history["train_loss"][-1],

        "final_val_loss":
            history["val_loss"][-1],

        "final_train_accuracy":
            history["train_accuracy"][-1],

        "final_val_accuracy":
            history["val_accuracy"][-1],

        "epochs":
            epochs,

        "optimizer":
            optimizer_name,

        "learning_rate":
            learning_rate,

        "batch_size":
            batch_size,

        "hidden_size":
            hidden_size,

        "dropout":
            dropout,

        "batch_norm":
            batch_norm
    }

    train_metrics_path = (
        metrics_dir /
        "train_metrics.json"
    )

    with open(
        train_metrics_path,
        "w"
    ) as file:

        json.dump(
            train_metrics,
            file,
            indent=4
        )

    # =====================================================
    # 18. FINAL MLFLOW METRIC
    # =====================================================

    mlflow.log_metric(
        "best_val_accuracy",
        best_val_accuracy
    )

    # =====================================================
    # 19. MLFLOW ARTIFACTS
    # =====================================================

    mlflow.log_artifact(
        str(best_checkpoint_path),
        artifact_path="checkpoints"
    )

    mlflow.log_artifact(
        str(final_model_path),
        artifact_path="models"
    )

    mlflow.log_artifact(
        str(train_metrics_path),
        artifact_path="metrics"
    )

    # =====================================================
    # 20. COMPLETE
    # =====================================================

    print(
        "\n========================================"
    )

    print(
        "Training completed successfully!"
    )

    print(
        "========================================"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best checkpoint: "
        f"{best_checkpoint_path}"
    )

    print(
        f"Final model: "
        f"{final_model_path}"
    )

    print(
        f"Checkpoint directory: "
        f"{checkpoint_dir}"
    )

    print(
        "\nMLflow run completed successfully!"
    )
