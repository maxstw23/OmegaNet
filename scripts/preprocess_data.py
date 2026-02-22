import uproot
import awkward as ak
import numpy as np
import torch
from tqdm import tqdm
import os


def run_balanced_preprocessing(input_file, output_file):
    # 1. Load the ROOT data
    print(f"Reading STAR TTree from {input_file}...")
    with uproot.open(input_file) as file:
        tree = file["ml_tree"]
        # Extract everything we need for kinematics and labels
        data = tree.arrays([
            "omega_px", "omega_py", "omega_pz", "omega_charge",
            "kaon_px", "kaon_py", "kaon_pz", "kaon_charge"
        ], library="ak")

    # 2. Build Global Pools for Sampling (The ``Blinding'' Strategy)
    # We flatten everything to get the global momentum distributions of K+ and K-
    print("Building global momentum pools for balancing...")
    all_k_px = ak.to_numpy(ak.flatten(data["kaon_px"]))
    all_k_py = ak.to_numpy(ak.flatten(data["kaon_py"]))
    all_k_pz = ak.to_numpy(ak.flatten(data["kaon_pz"]))
    all_k_q = ak.to_numpy(ak.flatten(data["kaon_charge"]))

    # Separate into Pos and Neg pools: Shape [N, 3]
    k_pos_pool = np.stack([all_k_px[all_k_q > 0], all_k_py[all_k_q > 0], all_k_pz[all_k_q > 0]], axis=1)
    k_neg_pool = np.stack([all_k_px[all_k_q < 0], all_k_py[all_k_q < 0], all_k_pz[all_k_q < 0]], axis=1)

    processed_graphs = []

    # 3. Process Events and Balance Counts
    print(f"Processing {len(data)} events...")
    for i in tqdm(range(len(data))):
        # Local Event Data
        k_px = ak.to_numpy(data[i]["kaon_px"])
        k_py = ak.to_numpy(data[i]["kaon_py"])
        k_pz = ak.to_numpy(data[i]["kaon_pz"])
        k_q = ak.to_numpy(data[i]["kaon_charge"])

        o_px = data[i]["omega_px"]
        o_py = data[i]["omega_py"]
        o_pz = data[i]["omega_pz"]
        o_charge = data[i]["omega_charge"]

        n_pos = np.sum(k_q > 0)
        n_neg = np.sum(k_q < 0)
        n_target = max(n_pos, n_neg)

        # Skip events with absolutely no kaons
        if n_target == 0:
            continue

        # Sample extra kaons from the global pool to reach n_target for both charges
        added_px, added_py, added_pz, added_q = [], [], [], []

        if n_neg < n_target:
            num_to_add = n_target - n_neg
            idx = np.random.choice(len(k_neg_pool), size=num_to_add, replace=True)
            samples = k_neg_pool[idx]
            added_px, added_py, added_pz = samples[:, 0], samples[:, 1], samples[:, 2]
            added_q = np.full(num_to_add, -1)

        elif n_pos < n_target:
            num_to_add = n_target - n_pos
            idx = np.random.choice(len(k_pos_pool), size=num_to_add, replace=True)
            samples = k_pos_pool[idx]
            added_px, added_py, added_pz = samples[:, 0], samples[:, 1], samples[:, 2]
            added_q = np.full(num_to_add, 1)

        # Concatenate Real + Fake
        f_px = np.concatenate([k_px, added_px])
        f_py = np.concatenate([k_py, added_py])
        f_pz = np.concatenate([k_pz, added_pz])
        f_q = np.concatenate([k_q, added_q])

        # 4. Feature Engineering
        # Calculate Transverse Momentum (pT)
        f_pt = np.sqrt(f_px ** 2 + f_py ** 2)
        # Calculate Momentum relative to the Omega candidate
        rel_px = f_px - o_px
        rel_py = f_py - o_py
        rel_pz = f_pz - o_pz

        # Construct Node Feature Matrix: [N_nodes, 5]
        # Features: [pT, rel_px, rel_py, rel_pz, charge]
        node_features = np.stack([f_pt, rel_px, rel_py, rel_pz, f_q], axis=1)

        # Label: Omega (-) = 0, Anti-Omega (+) = 1
        y_label = 1 if o_charge > 0 else 0

        # Convert to PyTorch Tensors
        processed_graphs.append({
            'x': torch.tensor(node_features, dtype=torch.float),
            'y': torch.tensor([y_label], dtype=torch.long)
        })

    # 5. Serialization
    print(f"Saving {len(processed_graphs)} graphs to {output_file}...")
    torch.save(processed_graphs, output_file)
    print("Done!")


if __name__ == "__main__":
    # Ensure directories exist
    if not os.path.exists("data"):
        os.makedirs("data")

    run_balanced_preprocessing(
        input_file="data/omega_kaon_all.root",
        output_file="data/balanced_omega_anti.pt"
    )