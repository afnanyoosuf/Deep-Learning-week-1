from pathlib import Path
import json

import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.pytorch_model import MNISTANN


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARAMS_PATH = PROJECT_ROOT / "params.yaml"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pytorch"
    / "mnist_ann.pth"
)

METRICS_DIR = PROJECT_ROOT / "metrics"

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_PATH = (
    METRICS_DIR / "test_metrics.json"
)


# ============================================================
# Load parameters
# ============================================================

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)


data_params = params["data"]
train_params = params["train"]
model_params = params["model"]


BATCH_SIZE = train_params["batch_size"]

INPUT_SIZE = model_params["input_size"]
HIDDEN_SIZE = model_params["hidden_size"]
OUTPUT_SIZE = model_params["output_size"]
DROPOUT = model_params["dropout"]
BATCH_NORM = model_params["batch_norm"]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Check model
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"""
Model not found:

{MODEL_PATH}

Run the training script first:

python src/training/train_pytorch.py
"""
    )


# ============================================================
# Test dataset
# ============================================================

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root=str(
        PROJECT_ROOT / "data" / "raw"
    ),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
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


# ============================================================
# Load trained weights
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# Evaluation
# ============================================================

correct = 0
total = 0

criterion = nn.CrossEntropyLoss()

total_loss = 0.0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


test_loss = (
    total_loss / len(test_loader)
)

test_accuracy = (
    correct / total
)


# ============================================================
# Save metrics
# ============================================================

metrics = {
    "test_loss": test_loss,
    "test_accuracy": test_accuracy
}


with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# Print results
# ============================================================

print()
print("==============================")
print("MNIST TEST RESULTS")
print("==============================")
print(
    f"Test Loss: {test_loss:.4f}"
)
print(
    f"Test Accuracy: {test_accuracy:.4f}"
)
print(
    f"Metrics saved: {METRICS_PATH}"
)
