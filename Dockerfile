FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1 && \
    pip install --no-cache-dir \
    numpy==2.3.5 \
    PyYAML==6.0.3 \
    Pillow==12.3.0 \
    mlflow==3.15.1

COPY . .

RUN mkdir -p \
    data/raw \
    data/processed \
    data/external \
    models/pytorch/checkpoints \
    metrics

CMD ["python", "-m", "src.training.train_pytorch"]