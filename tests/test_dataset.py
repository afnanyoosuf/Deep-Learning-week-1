import torch

from src.data.dataset import get_mnist_dataloaders


def test_mnist_dataloader():

    train_loader, val_loader, test_loader = (
        get_mnist_dataloaders(
            data_dir="data/raw",
            batch_size=32,
            train_size=50000,
            val_size=10000,
            seed=42
        )
    )

    images, labels = next(
        iter(train_loader)
    )

    assert images.shape[1:] == (
        1,
        28,
        28
    )

    assert labels.ndim == 1

    assert images.shape[0] <= 32
