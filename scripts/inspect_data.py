import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from tqdm import tqdm
import config


def analyze_features():
    print(f"Loading data from: {config.DATA_PATH}")
    # Load the new dataset generated with pT, eta, phi
    raw_data = torch.load(config.DATA_PATH)

    # Collect all node features (x)
    # New Shape: [f_pt, d_pt, d_eta, d_phi, f_q]
    all_features = []
    for entry in tqdm(raw_data, desc="Collecting features"):
        all_features.append(entry['x'][:, config.FEATURE_IDX])

    all_features = torch.cat(all_features, dim=0).numpy()

    feature_names = config.FEATURE_NAMES

    print("\n" + "=" * 65)
    print(f"{'Feature':<10} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 65)

    for i, name in enumerate(feature_names):
        col = all_features[:, i]
        print(f"{name:<10} | {np.mean(col):.3f} | {np.std(col):.3f} | {np.min(col):.3f} | {np.max(col):.3f}")
    print("=" * 65)

    # Sanity Check for d_phi wrapping
    if "d_phi" in config.FEATURE_NAMES:
        d_phi_col = all_features[:, config.FEATURE_NAMES.index("d_phi")]
        if np.max(d_phi_col) > np.pi or np.min(d_phi_col) < 0:
            print("!!! WARNING: |d_phi| values out of [0, pi] range !!!")

    if np.isnan(all_features).any():
        print("!!! WARNING: NaNs detected in dataset !!!")


if __name__ == "__main__":
    analyze_features()