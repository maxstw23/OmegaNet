import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
from torch.utils.data import DataLoader
import random
import config
from transformer_model import OmegaTransformer
from tqdm import tqdm


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    return padded, torch.stack(ys), mask


def evaluate():
    device = config.DEVICE
    print(f"Loading OmegaTransformer: {config.MODEL_SAVE_PATH}")

    model = OmegaTransformer(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    print("Preparing evaluation dataset...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds
        dataset.append((x, y.squeeze().long()))

    # Exact same 80/20 split as training
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    val_loader = DataLoader(dataset[split_idx:], batch_size=config.BATCH_SIZE, collate_fn=collate_fn)

    all_p_anti = []
    all_labels = []
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, mask)
            p_anti = torch.softmax(logits, dim=1)[:, 1]
            all_p_anti.append(p_anti.cpu())
            all_labels.append(y.cpu())

    p_anti = torch.cat(all_p_anti)
    labels = torch.cat(all_labels)
    is_omega = (labels == 0)
    is_anti = (labels == 1)
    n_omega = is_omega.sum().item()
    n_anti = is_anti.sum().item()
    # r = N_Anti / N_Omega in this dataset (used for f_BN correction)
    r_anti_omega = n_anti / n_omega

    # --- Argmax result (threshold = 0.5) ---
    preds = (p_anti >= 0.5).long()
    o_rec = ((preds == 0) & is_omega).sum().item() / n_omega
    a_rec = ((preds == 1) & is_anti).sum().item() / n_anti
    score = o_rec + a_rec - 1.0
    # f_BN correction: Omega_rec = f_BN + (1 - Anti_rec) * r  → f_BN = Omega_rec - (1-Anti_rec)*r
    f_bn = o_rec - (1.0 - a_rec) * r_anti_omega

    print("\n" + "=" * 50)
    print(f"  Argmax (threshold = 0.50)")
    print(f"  Omega Recall (raw):       {o_rec:.4f}")
    print(f"   Anti Recall:             {a_rec:.4f}")
    print(f" Physics Score:             {score:.4f}")
    print(f"  f_BN (corrected):         {f_bn:.4f}  [r={r_anti_omega:.3f}]")
    print("=" * 50)

    # --- Threshold scan ---
    print("\nThreshold scan (Anti recall → 1.0):")
    print(f"{'Threshold':>10} | {'Anti Recall':>11} | {'Omega Recall':>12} | {'Score':>7} | {'f_BN':>6}")
    print("-" * 60)
    thresholds = [t / 100 for t in range(10, 96, 5)]
    best_score_t, best_row = -1.0, None
    for t in thresholds:
        preds_t = (p_anti >= t).long()
        o_r = ((preds_t == 0) & is_omega).sum().item() / n_omega
        a_r = ((preds_t == 1) & is_anti).sum().item() / n_anti
        s = o_r + a_r - 1.0
        f = o_r - (1.0 - a_r) * r_anti_omega
        marker = " ← best score" if s > best_score_t else ""
        if s > best_score_t:
            best_score_t = s
            best_row = (t, a_r, o_r, s, f)
        print(f"{t:>10.2f} | {a_r:>11.4f} | {o_r:>12.4f} | {s:>7.4f} | {f:>6.4f}{marker}")
    if best_row:
        print(f"\nPhysics optimum: t={best_row[0]:.2f} | Anti={best_row[1]:.4f} | Omega={best_row[2]:.4f} | Score={best_row[3]:.4f} | f_BN={best_row[4]:.4f}")


if __name__ == "__main__":
    evaluate()
