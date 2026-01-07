#! pip install torch_geometric
#! pip install torch_scatter
import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean

class TrustGNN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(aggr='mean')

        self.lin_self = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_neigh = torch.nn.Linear(in_channels, hidden_channels)

        # Learn how much to trust self vs neighbors
        self.trust_gate = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
            torch.nn.Sigmoid()
        )

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h_self = self.lin_self(x)
        h_neigh = self.propagate(edge_index, x=x)

        gate = self.trust_gate(torch.cat([h_self, h_neigh], dim=1))
        h = gate * h_self + (1 - gate) * h_neigh

        return self.classifier(h)

    def message(self, x_j):
        return self.lin_neigh(x_j)
