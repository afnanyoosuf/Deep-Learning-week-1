import torch

from src.models.pytorch_model import MNISTANN


def test_model_output_shape():

    model = MNISTANN(
        input_size=784,
        hidden_size=128,
        output_size=10,
        dropout=0.3,
        batch_norm=True
    )

    x = torch.randn(4, 1, 28, 28)

    output = model(x)

    assert output.shape == (4, 10)
