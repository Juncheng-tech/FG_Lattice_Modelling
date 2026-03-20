import torch
import torch.nn as nn

# MLP model with flexible hidden size and activation function
# For comparison experiments in Week 6
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50, activation_name="relu"):
        super().__init__()

        # Supported activation functions
        activation_dict = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }

        # Check if the chosen activation is valid
        if activation_name not in activation_dict:
            raise ValueError(f"Unsupported activation: {activation_name}")

        activation = activation_dict[activation_name]

        # Build a 3-layer MLP structure
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Forward pass
        return self.network(x)