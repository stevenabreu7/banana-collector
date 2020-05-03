import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_units):
        """Initialize parameters and build model.
        Params
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_units (list(int)): Number of nodes in each hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.n_layers = len(hidden_units)
        self.in_layer = nn.Linear(state_size, hidden_units[0])
        self.hid_layers = [
            nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(self.n_layers-1)
        ]
        self.out_layer = nn.Linear(hidden_units[-1], action_size)

    def forward(self, state):
        """Forward pass that maps state to action values."""
        x = F.relu(self.in_layer(state))
        for hid_layer in self.hid_layers:
            x = F.relu(hid_layer(x))
        return self.out_layer(x)
