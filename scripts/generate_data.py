import torch
import numpy as np
from torch_geometric.data import Data
from torch_cluster import radius_graph
import os
import config


def sample_kaons(count, sigma, rel_q_value):
    """
    Generates relative coordinates for a specific number of kaons.
    Uses a Gaussian distribution centered at (0,0).
    """
    if count <= 0:
        return []
    d_eta = np.random.normal(0, sigma, count)
    d_phi = np.random.normal(0, sigma, count)

    # Periodic wrap for phi to stay within [-pi, pi]
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi

    return [[d_eta[i], d_phi[i], rel_q_value] for i in range(count)]


def generate_and_save_data():
    data_list = []

    # Physics definitions based on your config
    TARGET_K_EACH = config.N_KPLUS_BG  # Every event will have 20 K+ and 20 K-

    for _ in range(config.TOTAL_EVENTS):
        # 1. Identity Setup
        is_anti_omega = np.random.choice([True, False])
        is_string = np.random.choice([True, False])

        # Trigger Node always at [0, 0] with relative charge +1
        nodes = [[0.0, 0.0, 1.0]]

        # Track absolute particle counts for this event
        # Anti-Omega (+): OS is K- (-1), SS is K+ (+1)
        # Omega (-):      OS is K+ (-1), SS is K- (+1)
        current_k_plus = 0
        current_k_minus = 0

        # 2. String Correlated Kaons (Narrow Gaussian)
        # These are always Opposite-Sign (OS) to the trigger (rel_q = -1.0)
        if is_string:
            n_corr = np.random.randint(1, 4)
            nodes.extend(sample_kaons(n_corr, config.CORR_WIDTH, -1.0))

            # Update absolute counts based on trigger species
            if is_anti_omega:
                current_k_minus += n_corr  # K- is OS for Anti-Omega
            else:
                current_k_plus += n_corr  # K+ is OS for Omega

        # 3. Background / Ghost Kaons (Broad Gaussian)
        # We fill the remainder to ensure exactly 20 K+ and 20 K-
        rem_k_plus = TARGET_K_EACH - current_k_plus
        rem_k_minus = TARGET_K_EACH - current_k_minus

        # Sample K+ for background
        # (rel_q is +1 if Anti-Omega, -1 if Omega)
        q_val_kplus = 1.0 if is_anti_omega else -1.0
        nodes.extend(sample_kaons(rem_k_plus, config.SIGMA_BG, q_val_kplus))

        # Sample K- for background
        # (rel_q is -1 if Anti-Omega, +1 if Omega)
        q_val_kminus = -1.0 if is_anti_omega else 1.0
        nodes.extend(sample_kaons(rem_k_minus, config.SIGMA_BG, q_val_kminus))

        # 4. Graph Construction
        x = torch.tensor(nodes, dtype=torch.float)

        # Build edges based on proximity (Radius: 0.8)
        edge_index = radius_graph(x[:, :2], r=config.GRAPH_RADIUS, loop=False)

        # Calculate Delta R for edge_attr
        row, col = edge_index
        dist = torch.sqrt(torch.sum((x[row, :2] - x[col, :2]) ** 2, dim=1))
        edge_attr = dist.view(-1, 1)

        # Labels
        y = torch.tensor([1 if is_anti_omega else 0], dtype=torch.long)
        mech = torch.tensor([1 if is_string else 0], dtype=torch.long)

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mechanism=mech))

    # Save to disk
    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
    torch.save(data_list, config.DATA_PATH)

    print(f"Generated {config.TOTAL_EVENTS} events.")
    print(f"Consistency Check: Every event has exactly {TARGET_K_EACH} K+ and {TARGET_K_EACH} K-.")


if __name__ == "__main__":
    generate_and_save_data()