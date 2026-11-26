from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(98, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.network(z).squeeze(-1)