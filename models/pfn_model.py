import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


class OmegaPFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(OmegaPFN, self).__init__()

        # Phi Network: Processes each Kaon individually
        # Maps raw features into a high dimensional latent space
        self.phi = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Rho Network: Processes the summed representation of the event
        # Classifies the event as Omega vs Anti-Omega
        self.rho = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, batch):
        # 1. Apply Phi to every single Kaon
        # x shape: [Total_Kaons_in_Batch, Features]
        h_kaons = self.phi(x)

        # 2. Permutation Invariant Pooling (Deep Sets Summation)
        # Sums the Kaon latent vectors belonging to the same event
        # h_event shape: [Batch_Size, hidden_channels]
        h_event = global_add_pool(h_kaons, batch)

        # 3. Apply Rho to the pooled event representation
        out = self.rho(h_event)

        return out