from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
import mlflow

from src.data.dataset import get_mnist_dataloaders
from src.models.pytorch_model import MNISTANN


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARAMS_PATH = PROJECT_ROOT / "params.yaml"

MODEL_DIR = PROJECT_ROOT / "models" / "pytorch"
METRICS_DIR = PROJECT_ROOT / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "mnist_ann.pth"
METRICS_PATH = METRICS_DIR / "train_metrics.json"


# ============================================================
# Load parameters
# ============================================================

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)


data_params = params["data"]
train_params = params["train"]
model_params = params["model"]


SEED = train_params["seed"]
BATCH_SIZE = train_params["batch_size"]
EPOCHS = train_params["epochs"]
LEARNING_RATE = train_params["learning_rate"]
OPTIMIZER_NAME = train_params["optimizer"]

INPUT_SIZE = model_params["input_size"]
HIDDEN_SIZE = model_params["hidden_size"]
OUTPUT_SIZE = model_params["output_size"]
DROPOUT = model_params["dropout"]
BATCH_NORM = model_params["batch_norm"]


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Data
# ============================================================

train_loader, val_loader, test_loader = (
    get_mnist_dataloaders(
        data_dir=PROJECT_ROOT / "data" / "raw",
        batch_size=BATCH_SIZE,
        train_size=data_params["train_size"],
        val_size=data_params["val_size"],
        seed=SEED
    )
)


# ============================================================
# Model
# ============================================================

model = MNISTANN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    dropout=DROPOUT,
    batch_norm=BATCH_NORM
).to(device)

print(model)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

if OPTIMIZER_NAME.lower() == "adam":

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

elif OPTIMIZER_NAME.lower() == "sgd":

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

elif OPTIMIZER_NAME.lower() == "rmsprop":

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=LEARNING_RATE
    )

else:

    raise ValueError(
        f"Unsupported optimizer: {OPTIMIZER_NAME}"
    )


# ============================================================
# MLflow
# ============================================================

mlflow.set_tracking_uri(
    "http://localhost:5000"
)

mlflow.set_experiment(
    "MNIST ANN DVC Experiments"
)


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


with mlflow.start_run():

    # Log parameters
    mlflow.log_params({
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER_NAME,
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "batch_norm": BATCH_NORM
    })

    for epoch in range(EPOCHS):

        # ====================================================
        # Training
        # ====================================================

        model.train()

        train_loss = 0.0
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

            train_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss /= len(train_loader)

        train_accuracy = (
            train_correct / train_total
        )


        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_loss = 0.0
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

                val_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= len(val_loader)

        val_accuracy = (
            val_correct / val_total
        )


        # ====================================================
        # Save history
        # ====================================================

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )


        # ====================================================
        # MLflow metrics
        # ====================================================

        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }, step=epoch)


        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )


        # ====================================================
        # Save best model
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )


    # ========================================================
    # Save final metrics
    # ========================================================

    metrics = {
        "best_val_accuracy": best_val_accuracy,
        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_accuracy": history["val_accuracy"][-1]
    }

    with open(METRICS_PATH, "w") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    mlflow.log_metrics(metrics)


print()
print("Training completed.")
print("Model saved to:", MODEL_PATH)
print("Metrics saved to:", METRICS_PATH)
