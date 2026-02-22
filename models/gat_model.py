import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool


class PhysicsGAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super(PhysicsGAT, self).__init__()
        # Input features: [k_pt, rel_px, rel_py, rel_pz, k_q]
        self.conv1 = GATv2Conv(5, hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1)

        # We concatenate Mean, Max, and Sum pooling, so the input to Linear is hidden_channels * 3
        self.lin = torch.nn.Linear(hidden_channels * 3, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Multi-descriptor pooling
        mean_p = global_mean_pool(x, batch)
        max_p = global_max_pool(x, batch)
        sum_p = global_add_pool(x, batch)

        # Combine the "average" Kaon, the "extreme" Kaon, and the "total" cloud
        combined = torch.cat([mean_p, max_p, sum_p], dim=1)

        combined = F.dropout(combined, p=0.3, training=self.training)
        return self.lin(combined)