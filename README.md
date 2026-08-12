# Deep Learning MNIST — PyTorch, DVC, MLflow and Docker

This project implements an end-to-end reproducible deep learning workflow for MNIST classification using PyTorch.

The project includes:

* PyTorch model training
* Batch Normalization
* Dropout regularization
* DVC pipeline management
* DVC metrics tracking
* DVC S3 remote storage
* MLflow experiment tracking
* Dockerized training environment
* GitHub Actions CI
* Reproducible model checkpoints

## Project Structure

```text
Project/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── raw/
├── metrics/
│   ├── train_metrics.json
│   └── test_metrics.json
├── models/
│   └── pytorch/
├── src/
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── models/
│   │   └── pytorch_model.py
│   ├── training/
│   │   └── train_pytorch.py
│   └── mlflow_config.py
├── Dockerfile
├── .dockerignore
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── requirements-docker.txt
```

## Model

The PyTorch model is a fully connected neural network for MNIST classification.

Architecture:

```text
Input: 784
   ↓
Linear(784 → 128)
   ↓
BatchNorm1d(128)
   ↓
ReLU
   ↓
Dropout(0.3)
   ↓
Linear(128 → 10)
   ↓
Output
```

Training configuration:

```text
Epochs:         10
Batch size:     64
Learning rate:  0.001
Optimizer:      Adam
Hidden size:    128
Dropout:        0.3
Batch norm:     True
Random seed:    42
```

## Results

The trained model achieved approximately:

```text
Best validation accuracy: 97.4%
Test accuracy:             97.68%
Test loss:                 0.079
```

The exact results can be inspected using:

```powershell
dvc metrics show
```

## DVC Pipeline

The pipeline contains two stages:

```text
data/raw
    ↓
train_pytorch
    ↓
best_checkpoint.pth
    ↓
evaluate_pytorch
    ↓
test_metrics.json
```

View the pipeline:

```powershell
dvc dag
```

Check pipeline status:

```powershell
dvc status
```

Reproduce the pipeline:

```powershell
dvc repro
```

Display metrics:

```powershell
dvc metrics show
```

Push DVC artifacts:

```powershell
dvc push
```

The project uses an S3 DVC remote for model/artifact storage.

## MLflow

MLflow is used to track:

* Training parameters
* Training metrics
* Validation metrics
* Model information
* Experiment runs

The experiment is:

```text
Deep Learning Experiments
```

Start the MLflow server locally:

```powershell
mlflow server --host 0.0.0.0 --port 5000
```

Then open:

```text
http://localhost:5000
```

For Docker training, the container connects to the host MLflow server using:

```text
http://host.docker.internal:5000
```

## Docker

The training environment is containerized using Docker.

Build the image:

```powershell
docker build -t deep-learning-pytorch:latest .
```

Verify PyTorch:

```powershell
docker run --rm deep-learning-pytorch:latest python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

Run training using the host project directory:

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -e "MLFLOW_TRACKING_URI=http://host.docker.internal:5000" `
  deep-learning-pytorch:latest `
  python -m src.training.train_pytorch
```

The current Docker training environment uses CPU PyTorch.

## Evaluation

Run evaluation from the project environment:

```powershell
python -m src.evaluation.evaluate
```

Or from Docker:

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -e "MLFLOW_TRACKING_URI=http://host.docker.internal:5000" `
  deep-learning-pytorch:latest `
  python -m src.evaluation.evaluate
```

## GitHub Actions

The repository contains a GitHub Actions workflow:

```text
.github/workflows/ci.yml
```

The workflow:

1. Checks out the repository
2. Installs Python 3.11
3. Installs the Docker/CI dependencies
4. Verifies PyTorch and Torchvision
5. Compiles the Python source files
6. Checks the DVC pipeline

The workflow runs automatically on pushes and pull requests targeting `main`.

## Reproducibility Workflow

The complete workflow is:

```text
GitHub
   ↓
Source code
   ↓
DVC pipeline
   ↓
Docker environment
   ↓
PyTorch training
   ↓
Model checkpoints
   ↓
MLflow experiment tracking
   ↓
DVC artifact storage
   ↓
Evaluation
   ↓
Metrics
```

## Development Commands

Check Git:

```powershell
git status
```

Check DVC:

```powershell
dvc status
```

Reproduce training:

```powershell
dvc repro
```

Show metrics:

```powershell
dvc metrics show
```

Push DVC artifacts:

```powershell
dvc push
```

Build Docker image:

```powershell
docker build -t deep-learning-pytorch:latest .
```

## Notes

Model `.pth` files are intentionally excluded from Git and are managed by DVC.

The MLflow tracking server is separate from the Docker training container. When running training inside Docker on Windows/Docker Desktop, `host.docker.internal` is used to reach an MLflow server running on the host machine.

Secrets such as AWS credentials must not be committed to GitHub.
