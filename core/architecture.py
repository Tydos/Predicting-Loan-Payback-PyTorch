import torch.nn as nn


class LoanPredictor(nn.Module):
    def __init__(self, num_features: int, hidden_layers: list, dropout: float):
        super().__init__()
        layers = []
        in_dim = num_features
        for out_dim in hidden_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
