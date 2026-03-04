import uproot
import awkward as ak
import numpy as np
import torch
from tqdm import tqdm
import os


def run_balanced_preprocessing(input_file, output_file):
    print(f"Reading STAR TTree from {input_file}...")
    with uproot.open(input_file) as file:
        tree = file["ml_tree"]
        data = tree.arrays([
            "omega_px", "omega_py", "omega_pz", "omega_charge",
            "kaon_px", "kaon_py", "kaon_pz", "kaon_charge"
        ], library="ak")

    print("Building global momentum pools for balancing...")
    all_k_px = ak.to_numpy(ak.flatten(data["kaon_px"]))
    all_k_py = ak.to_numpy(ak.flatten(data["kaon_py"]))
    all_k_pz = ak.to_numpy(ak.flatten(data["kaon_pz"]))
    all_k_q = ak.to_numpy(ak.flatten(data["kaon_charge"]))

    k_pos_pool = np.stack([all_k_px[all_k_q > 0], all_k_py[all_k_q > 0], all_k_pz[all_k_q > 0]], axis=1)
    k_neg_pool = np.stack([all_k_px[all_k_q < 0], all_k_py[all_k_q < 0], all_k_pz[all_k_q < 0]], axis=1)

    processed_graphs = []

    print(f"Processing {len(data)} events...")
    for i in tqdm(range(len(data))):
        k_px = ak.to_numpy(data[i]["kaon_px"])
        k_py = ak.to_numpy(data[i]["kaon_py"])
        k_pz = ak.to_numpy(data[i]["kaon_pz"])
        k_q = ak.to_numpy(data[i]["kaon_charge"])

        o_px, o_py, o_pz = data[i]["omega_px"], data[i]["omega_py"], data[i]["omega_pz"]
        o_charge = data[i]["omega_charge"]

        n_pos, n_neg = np.sum(k_q > 0), np.sum(k_q < 0)
        n_target = max(n_pos, n_neg)
        if n_target == 0: continue

        # Balancing logic
        added_px, added_py, added_pz, added_q = [], [], [], []
        if n_neg < n_target:
            idx = np.random.choice(len(k_neg_pool), size=n_target - n_neg, replace=True)
            samples = k_neg_pool[idx]
            added_px, added_py, added_pz, added_q = samples[:, 0], samples[:, 1], samples[:, 2], np.full(
                n_target - n_neg, -1)
        elif n_pos < n_target:
            idx = np.random.choice(len(k_pos_pool), size=n_target - n_pos, replace=True)
            samples = k_pos_pool[idx]
            added_px, added_py, added_pz, added_q = samples[:, 0], samples[:, 1], samples[:, 2], np.full(
                n_target - n_pos, 1)

        f_px = np.concatenate([k_px, added_px])
        f_py = np.concatenate([k_py, added_py])
        f_pz = np.concatenate([k_pz, added_pz])
        f_q = np.concatenate([k_q, added_q])

        # --- COORDINATE TRANSFORMATION ---
        # 1. Omega Kinematics
        o_pt = np.sqrt(o_px ** 2 + o_py ** 2)
        o_p = np.sqrt(o_px ** 2 + o_py ** 2 + o_pz ** 2)
        o_eta = np.arctanh(o_pz / o_p)
        o_phi = np.arctan2(o_py, o_px)

        # 2. Kaon Kinematics
        f_pt = np.sqrt(f_px ** 2 + f_py ** 2)
        f_p = np.sqrt(f_px ** 2 + f_py ** 2 + f_pz ** 2)
        f_eta = np.arctanh(f_pz / f_p)
        f_phi = np.arctan2(f_py, f_px)

        # 3. Relative Features (Delta Eta and Delta Phi)
        d_ptx = f_px - o_px
        d_pty = f_py - o_py
        d_pt = np.sqrt(d_ptx ** 2 + d_pty ** 2)
        d_eta = f_eta - o_eta
        d_phi = f_phi - o_phi

        # Handle Phi wrapping (ensure value is between -pi and pi)
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi

        # Node Features: [pT, d_pt, d_eta, d_phi, charge]
        # f_q encoded as relative sign w.r.t. Omega: same-sign = +1, opposite-sign = -1
        rel_q = f_q * np.sign(o_charge)
        node_features = np.stack([f_pt, d_pt, d_eta, d_phi, rel_q], axis=1)

        y_label = 1 if o_charge > 0 else 0
        processed_graphs.append({
            'x': torch.tensor(node_features, dtype=torch.float),
            'y': torch.tensor([y_label], dtype=torch.long)
        })

    print(f"Saving {len(processed_graphs)} graphs...")
    torch.save(processed_graphs, output_file)

    print("Computing feature statistics...")
    all_features = torch.cat([g['x'] for g in processed_graphs], dim=0)
    means = all_features.mean(dim=0)
    stds = all_features.std(dim=0)
    stds[stds == 0] = 1.0  # avoid division by zero for constant features (e.g. f_q after balancing)
    stats_file = output_file.replace(".pt", "_stats.pt")
    torch.save({'means': means, 'stds': stds}, stats_file)
    print(f"Feature stats saved to {stats_file}")
    for name, m, s in zip(["f_pt", "d_pt", "d_eta", "d_phi", "f_q"], means, stds):
        print(f"  {name}: mean={m:.4f}, std={s:.4f}")
    print("Done!")


if __name__ == "__main__":
    # Ensure directories exist
    if not os.path.exists("data"):
        os.makedirs("data")

    run_balanced_preprocessing(
        input_file="data/omega_kaon_all.root",
        output_file="data/balanced_omega_anti.pt"
    )