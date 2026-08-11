import torch.nn as nn


class MNISTANN(nn.Module):

    def __init__(
        self,
        input_size=784,
        hidden_size=128,
        output_size=10,
        dropout=0.3,
        batch_norm=True
    ):
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(input_size, hidden_size)
        ]

        if batch_norm:
            layers.append(
                nn.BatchNorm1d(hidden_size)
            )

        layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(
                nn.Dropout(dropout)
            )

        layers.append(
            nn.Linear(hidden_size, output_size)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
