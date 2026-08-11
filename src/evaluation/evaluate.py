import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.pytorch_model import MNISTANN


# =========================================================
# 1. LOAD PARAMETERS
# =========================================================

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)


batch_size = params["train"]["batch_size"]

hidden_size = params["model"]["hidden_size"]
dropout = params["model"]["dropout"]
batch_norm = params["model"]["batch_norm"]


# =========================================================
# 2. DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================================================
# 3. PATHS
# =========================================================

checkpoint_path = Path(
    "models/pytorch/best_checkpoint.pth"
)

metrics_dir = Path("metrics")

metrics_dir.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 4. CHECK CHECKPOINT EXISTS
# =========================================================

if not checkpoint_path.exists():

    raise FileNotFoundError(
        f"Checkpoint not found: {checkpoint_path}"
    )


print(
    f"Loading checkpoint: {checkpoint_path}"
)


# =========================================================
# 5. LOAD MNIST TEST DATASET
# =========================================================

transform = transforms.ToTensor()


test_dataset = datasets.MNIST(
    root="data/raw",
    train=False,
    download=True,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


print(
    f"Test samples: {len(test_dataset)}"
)


# =========================================================
# 6. CREATE MODEL
# =========================================================

model = MNISTANN(
    hidden_size=hidden_size,
    dropout=dropout,
    batch_norm=batch_norm
).to(device)


# =========================================================
# 7. LOAD BEST CHECKPOINT
# =========================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


print(
    f"Loaded checkpoint from epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)


# =========================================================
# 8. LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# 9. EVALUATION
# =========================================================

model.eval()

running_test_loss = 0.0

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        running_test_loss += loss.item()


        # Predictions
        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


# =========================================================
# 10. CALCULATE TEST METRICS
# =========================================================

test_loss = (
    running_test_loss /
    len(test_loader)
)


test_accuracy = (
    correct /
    total
)


# =========================================================
# 11. SAVE TEST METRICS
# =========================================================

test_metrics = {

    "test_loss":
        test_loss,

    "test_accuracy":
        test_accuracy,

    "test_samples":
        total,

    "checkpoint_epoch":
        checkpoint["epoch"],

    "best_validation_accuracy":
        checkpoint["val_accuracy"],

    "optimizer":
        checkpoint["optimizer"],

    "learning_rate":
        checkpoint["learning_rate"],

    "batch_size":
        checkpoint["batch_size"],

    "hidden_size":
        checkpoint["hidden_size"],

    "dropout":
        checkpoint["dropout"],

    "batch_norm":
        checkpoint["batch_norm"]
}


test_metrics_path = (
    metrics_dir / "test_metrics.json"
)


with open(
    test_metrics_path,
    "w"
) as file:

    json.dump(
        test_metrics,
        file,
        indent=4
    )


# =========================================================
# 12. PRINT RESULTS
# =========================================================

print("\n========================================")
print("TEST EVALUATION COMPLETE")
print("========================================")

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Checkpoint Epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Validation Accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

print(
    f"Test Metrics: "
    f"{test_metrics_path}"
)
