# PyTorch ANN - MNIST

A reproducible PyTorch Artificial Neural Network workflow for MNIST digit classification.

## Project Overview

This project implements a complete Deep Learning workflow using:

- PyTorch
- MNIST
- DVC
- MLflow
- Docker
- GitHub Actions

The project covers dataset loading, model development, training, validation, optimization experiments, checkpointing, experiment tracking, data/model versioning, and containerization.

## Model Architecture

Input Image
→ Flatten
→ Linear (784 → 128)
→ Batch Normalization
→ ReLU
→ Dropout (0.3)
→ Linear (128 → 10)

## Experiments

The project evaluates:

- SGD
- SGD + Momentum
- Adam
- RMSprop
- Multiple learning rates
- Dropout
- Batch Normalization
- Training/validation convergence

## Project Structure

```text
deep-learning/
├── data/
├── models/
├── notebooks/
├── src/
├── tests/
├── .github/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
