"""plot_paper_figures.py — Generate paper-quality figures for OmegaNet.

Figures produced:
  1. plots/paper_score_dist.png         — Score distribution (Ω̄⁺ vs Ω⁻, val set)
  2. plots/paper_closure_test.png       — Closure test (Ω̄⁺ train vs val)
  3. plots/paper_purity_vs_cut.png      — Purity + FOM vs score threshold
  4. plots/paper_bias_check.png         — Score vs Ω kinematics (pT, |y|)
  5. plots/paper_omega_pt.png           — Ω pT for BN/PP subpops + Ω̄⁺ reference
  6. plots/paper_kaon_agg_bias.png      — Score vs kaon aggregate kinematics
  7. plots/feature_importance.png       — Permutation importance (--permtest, via interpret_model)

Usage:
  venv/bin/python scripts/plot_paper_figures.py                    # loads cache
  venv/bin/python scripts/plot_paper_figures.py --rescore          # force re-score
  venv/bin/python scripts/plot_paper_figures.py --permtest         # + permutation importance
"""
import sys, os, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import config
from transformer_model import OmegaTransformer
from tqdm import tqdm


def _manual_auc(y_true, scores):
    """Compute ROC AUC without sklearn."""
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    desc = np.argsort(-scores)
    y_sorted = y_true[desc]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = np.cumsum(y_sorted)
    fp = np.arange(1, len(y_sorted) + 1) - tp
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapezoid(tpr, fpr))

CACHE_PATH = "data/paper_scores.pt"


# ── Dataset helpers (identical to analyze_subpopulations.py) ──────────────────

class KaonDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate_fn(batch):
    xs, ys, raw_xs = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat  = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask   = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n]   = False
    return padded, torch.stack(ys), mask, list(raw_xs)


# ── Scoring pass ──────────────────────────────────────────────────────────────

def score_all_events():
    """Load model + data, score every event, cache results."""
    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    model = OmegaTransformer(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    stats = torch.load(config.STATS_PATH_UNPADDED)
    means = stats['means'][config.FEATURE_IDX]
    stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH_UNPADDED)
    dataset = []
    print("Preparing dataset...")
    for entry in tqdm(raw_data):
        raw_x = entry['x']
        y = entry['y'].squeeze().long()
        x = raw_x[:, config.FEATURE_IDX].clone()
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - means) / stds
        dataset.append((x, y, raw_x))

    loader = DataLoader(KaonDataset(dataset), batch_size=256,
                        collate_fn=collate_fn, num_workers=0)

    all_p, all_y, all_raw = [], [], []
    print("Scoring events...")
    with torch.no_grad():
        for x, y, mask, raw_xs in loader:
            x, mask = x.to(config.DEVICE), mask.to(config.DEVICE)
            p = torch.softmax(model(x, mask), dim=1)[:, 1].cpu()
            all_p.append(p); all_y.append(y); all_raw.extend(raw_xs)

    p_all = torch.cat(all_p)
    y_all = torch.cat(all_y)

    # Reproduce train/val split (same seed + shuffle as train.py / evaluate_physics.py)
    n = len(dataset)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * n)
    val_indices = set(indices[split:])
    val_mask = torch.tensor([i in val_indices for i in range(n)], dtype=torch.bool)

    os.makedirs("data", exist_ok=True)
    torch.save({'p_all': p_all, 'y_all': y_all, 'raw_all': all_raw, 'val_mask': val_mask},
               CACHE_PATH)
    print(f"Saved cache → {CACHE_PATH}")
    return p_all, y_all, all_raw, val_mask


def load_or_score(rescore=False):
    if not rescore and os.path.exists(CACHE_PATH):
        print(f"Loading cached scores from {CACHE_PATH}...")
        d = torch.load(CACHE_PATH)
        return d['p_all'], d['y_all'], d['raw_all'], d['val_mask']
    return score_all_events()


# ── Operating-point sweep (inline; no import from analyze_subpopulations) ─────

def compute_cutoffs(p_omega_np, p_anti_np, pi, min_frac=0.05, max_frac=0.65):
    """Return (opt_bn, opt_pp) tuples: (frac, threshold, purity, N, fom)."""
    n_O = len(p_omega_np)
    fracs = np.linspace(min_frac, max_frac, 200)
    bn_rows, pp_rows = [], []
    for frac in fracs:
        t_bn = np.quantile(p_omega_np, frac)
        f_O_bn = (p_omega_np < t_bn).mean()
        f_A_bn = (p_anti_np  < t_bn).mean()
        bn_pur = (f_O_bn - pi * f_A_bn) / f_O_bn if f_O_bn > 0 else 0.0
        n_bn   = int(f_O_bn * n_O)
        fom_bn = bn_pur * np.sqrt(n_bn) if bn_pur > 0 else 0.0
        bn_rows.append((frac, t_bn, bn_pur, n_bn, fom_bn))

        t_pp = np.quantile(p_omega_np, 1.0 - frac)
        f_O_pp = (p_omega_np >= t_pp).mean()
        f_A_pp = (p_anti_np  >= t_pp).mean()
        pp_pur = min(pi * f_A_pp / f_O_pp, 1.0) if f_O_pp > 0 else 0.0
        n_pp   = int(f_O_pp * n_O)
        fom_pp = pp_pur * np.sqrt(n_pp) if pp_pur > 0 else 0.0
        pp_rows.append((frac, t_pp, pp_pur, n_pp, fom_pp))

    bn_arr = np.array(bn_rows)
    pp_arr = np.array(pp_rows)
    opt_bn = bn_arr[np.argmax(bn_arr[:, 4])]
    opt_pp = pp_arr[np.argmax(pp_arr[:, 4])]
    return opt_bn, opt_pp, bn_arr, pp_arr


# ── Figure 1: Score distribution ──────────────────────────────────────────────

def plot_score_dist(p_all, y_all, val_mask):
    is_omega = (y_all == 0)
    is_anti  = (y_all == 1)
    p_val_omega = p_all[val_mask & is_omega].numpy()
    p_val_anti  = p_all[val_mask & is_anti].numpy()
    y_val = y_all[val_mask].numpy()
    p_val = p_all[val_mask].numpy()
    auc = _manual_auc(y_val, p_val)

    bins = np.linspace(0, 1, 51)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p_val_anti,  bins=bins, density=True, histtype='step', color='#2ca02c',
            linewidth=1.5, label='Ω̄⁺ (reference, y=1)')
    ax.hist(p_val_omega, bins=bins, density=True, histtype='step', color='#d62728',
            linewidth=1.5, label='Ω⁻ (mixture, y=0)')
    ax.set_xlabel('p(Anti)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score distribution (validation set)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'AUC = {auc:.4f}', transform=ax.transAxes,
            fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    out = 'plots/paper_score_dist.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ── Figure 2: Closure test ────────────────────────────────────────────────────

def plot_closure_test(p_all, y_all, val_mask):
    is_anti = (y_all == 1)
    anti_train = is_anti & ~val_mask
    anti_val   = is_anti &  val_mask
    p_train = p_all[anti_train].numpy()
    p_val   = p_all[anti_val].numpy()
    n_train, n_val = len(p_train), len(p_val)

    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p_train, bins=bins, density=True, histtype='step', color='#1f77b4',
            linewidth=1.5, linestyle='--', label=f'Ω̄⁺ train (N={n_train})')
    ax.hist(p_val,   bins=bins, density=True, histtype='step', color='#ff7f0e',
            linewidth=1.5, linestyle='-',  label=f'Ω̄⁺ val (N={n_val})')
    ax.set_xlabel('p(Anti)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Closure test: Ω̄⁺ train vs val', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = 'plots/paper_closure_test.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ── Figure 3: Purity vs cut + FOM ────────────────────────────────────────────

def plot_purity_vs_cut(p_all, y_all):
    is_omega = (y_all == 0)
    is_anti  = (y_all == 1)
    pi = is_anti.float().sum().item() / is_omega.float().sum().item()
    p_omega_np = p_all[is_omega].numpy()
    p_anti_np  = p_all[is_anti].numpy()

    opt_bn, opt_pp, bn_arr, pp_arr = compute_cutoffs(p_omega_np, p_anti_np, pi)

    # Convert percentile arrays to threshold space for x-axis
    thresholds_bn = bn_arr[:, 1]
    thresholds_pp = pp_arr[:, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=False)

    # Top panel: purity vs threshold
    ax1.plot(thresholds_bn, bn_arr[:, 2], color='#d62728', label='BN purity')
    ax1.plot(thresholds_pp, pp_arr[:, 2], color='#1f77b4', label='PP purity')
    ax1.axvline(opt_bn[1], color='#d62728', ls='--', alpha=0.7,
                label=f'BN opt t={opt_bn[1]:.3f}')
    ax1.axvline(opt_pp[1], color='#1f77b4', ls='--', alpha=0.7,
                label=f'PP opt t={opt_pp[1]:.3f}')
    ax1.set_ylabel('Mechanism purity', fontsize=11)
    ax1.set_title('Purity and FOM vs score threshold', fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: FOM vs threshold
    ax2.plot(thresholds_bn, bn_arr[:, 4], color='#d62728', label='FOM (BN)')
    ax2.plot(thresholds_pp, pp_arr[:, 4], color='#1f77b4', label='FOM (PP)')
    ax2.plot(opt_bn[1], opt_bn[4], '*', color='#d62728', ms=12)
    ax2.plot(opt_pp[1], opt_pp[4], '*', color='#1f77b4', ms=12)
    ax2.set_xlabel('Score threshold', fontsize=11)
    ax2.set_ylabel('FOM = purity × √N', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'plots/paper_purity_vs_cut.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")
    return float(opt_bn[1]), float(opt_pp[1]), float(opt_bn[2]), float(opt_pp[2])


# ── Figure 4: Bias check ──────────────────────────────────────────────────────

def plot_bias_check(p_all, y_all, raw_all):
    is_omega = (y_all == 0)
    is_anti  = (y_all == 1)

    # Extract per-event scalar features from kaon index 0 (broadcast features)
    o_pt_all = torch.tensor([r[0, 4].item()  for r in raw_all])   # o_pt  (index 4)
    o_y_all  = torch.tensor([r[0, 11].item() for r in raw_all])   # o_y_abs (index 11)

    pt_edges = np.linspace(0, 3, 11)
    y_edges  = np.linspace(0, 1.5, 11)

    def bin_mean_std(scores, feature, edges):
        means, errs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (feature >= lo) & (feature < hi)
            s = scores[sel].numpy()
            if len(s) < 2:
                means.append(np.nan); errs.append(np.nan)
            else:
                means.append(s.mean())
                errs.append(s.std() / np.sqrt(len(s)))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, np.array(means), np.array(errs)

    p_omega = p_all[is_omega]
    p_anti  = p_all[is_anti]
    pt_omega = o_pt_all[is_omega]; pt_anti = o_pt_all[is_anti]
    y_omega  = o_y_all[is_omega];  y_anti  = o_y_all[is_anti]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    panels = [
        (axes[0, 0], p_omega, pt_omega, pt_edges, 'Ω⁻ mean score vs pT', 'pT [GeV/c]'),
        (axes[0, 1], p_omega, y_omega,  y_edges,  'Ω⁻ mean score vs |y|', '|y_Ω|'),
        (axes[1, 0], p_anti,  pt_anti,  pt_edges, 'Ω̄⁺ mean score vs pT', 'pT [GeV/c]'),
        (axes[1, 1], p_anti,  y_anti,   y_edges,  'Ω̄⁺ mean score vs |y|', '|y_Ω|'),
    ]

    for ax, scores, feat, edges, title, xlabel in panels:
        centers, means, errs = bin_mean_std(scores, feat, edges)
        global_mean = scores.numpy().mean()
        ax.errorbar(centers, means, yerr=errs, fmt='o-', capsize=3, linewidth=1.5)
        ax.axhline(global_mean, color='gray', ls='--', alpha=0.7, label=f'Global mean = {global_mean:.3f}')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Mean p(Anti)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Score vs Ω kinematics (bias check)', fontsize=13)
    plt.tight_layout()
    out = 'plots/paper_bias_check.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ── Figure 5: Ω pT subpopulation comparison ───────────────────────────────────

def plot_omega_pt(p_all, y_all, raw_all, lo_thresh, hi_thresh, bn_purity, pp_purity):
    is_omega = (y_all == 0)
    is_anti  = (y_all == 1)

    bn_mask = is_omega & (p_all < lo_thresh)
    pp_mask = is_omega & (p_all >= hi_thresh)

    def get_pt(mask):
        idxs = torch.where(mask)[0]
        return np.array([raw_all[i.item()][0, 4].item() for i in idxs])

    pt_bn   = get_pt(bn_mask)
    pt_pp   = get_pt(pp_mask)
    pt_anti = get_pt(is_anti)

    bins = np.linspace(0, 3, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def normed_step(vals, bins):
        counts, _ = np.histogram(np.clip(vals, bins[0], bins[-1]), bins=bins)
        norm = counts.sum() * (bins[1] - bins[0])
        return counts / norm if norm > 0 else counts

    fig, ax = plt.subplots(figsize=(6, 4))
    for vals, color, label in [
        (pt_bn,   '#d62728', f'BN-enriched Ω⁻ (BN pur={bn_purity:.2f})'),
        (pt_pp,   '#1f77b4', f'PP-enriched Ω⁻ (PP pur={pp_purity:.2f})'),
        (pt_anti, '#2ca02c', 'Ω̄⁺ reference'),
    ]:
        ax.step(centers, normed_step(vals, bins), where='mid',
                color=color, linewidth=1.5, label=label)

    ax.set_xlabel('Ω pT [GeV/c]', fontsize=12)
    ax.set_ylabel('dN/dpT [a.u.]', fontsize=12)
    ax.set_title('Ω pT: BN-enriched vs PP-enriched vs Ω̄⁺', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = 'plots/paper_omega_pt.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ── Figure 6: Kaon aggregate kinematics bias ──────────────────────────────────

def plot_kaon_aggregate_bias(p_all, y_all, raw_all):
    """Score vs per-event aggregate kaon kinematics.

    Panels: mean |y_K − y_Ω|, mean |y_K|−|y_Ω|, mean kaon pT,
            mean kaon rapidity y_K, net_kaon (K⁺−K⁻).
    Ω⁻ and Ω̄⁺ overlaid for direct comparison.

    Raw FEATURE_REGISTRY indices used:
      0: f_pt,  2: d_y (|y_K − y_Ω|),  10: d_y_signed (|y_K|−|y_Ω|),
      12: net_kaon (broadcast, same for all kaons in event),  14: f_y
    """
    is_omega = (y_all == 0)
    is_anti  = (y_all == 1)

    net_kaon    = torch.tensor([r[0, 12].item()        for r in raw_all], dtype=torch.float32)
    mean_fpt    = torch.tensor([r[:, 0].mean().item()  for r in raw_all])
    mean_dy     = torch.tensor([r[:, 2].mean().item()  for r in raw_all])
    mean_dy_sgn = torch.tensor([r[:, 10].mean().item() for r in raw_all])
    mean_fy     = torch.tensor([r[:, 14].mean().item() for r in raw_all])

    def bin_mean_std(scores, feature, edges):
        means, errs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (feature >= lo) & (feature < hi)
            s = scores[sel].numpy()
            if len(s) < 2:
                means.append(np.nan); errs.append(np.nan)
            else:
                means.append(s.mean())
                errs.append(s.std() / np.sqrt(len(s)))
        return 0.5 * (edges[:-1] + edges[1:]), np.array(means), np.array(errs)

    dy_edges    = np.linspace(0,    2.0, 11)
    dysgn_edges = np.linspace(-1.5, 1.5, 11)
    fpt_edges   = np.linspace(0,    3.0, 11)
    fy_edges    = np.linspace(-1.5, 1.5, 11)
    nk_edges    = np.arange(-4, 10) - 0.5   # net_kaon: K⁺−K⁻

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    panels = [
        (axes[0, 0], mean_dy,     dy_edges,    'Score vs mean |y_K − y_Ω|',   'mean |d_y| per event'),
        (axes[0, 1], mean_dy_sgn, dysgn_edges, 'Score vs mean |y_K| − |y_Ω|', 'mean d_y_signed per event'),
        (axes[0, 2], mean_fpt,    fpt_edges,   'Score vs mean kaon pT',        'mean kaon pT [GeV/c]'),
        (axes[1, 0], mean_fy,     fy_edges,    'Score vs mean kaon rapidity',  'mean y_K per event'),
        (axes[1, 1], net_kaon,    nk_edges,    'Score vs net kaon',            'net_kaon (K⁺ − K⁻)'),
    ]
    axes[1, 2].set_visible(False)

    for ax, feat_all, edges, title, xlabel in panels:
        for cls_mask, color, label in [
            (is_omega, '#d62728', 'Ω⁻'),
            (is_anti,  '#2ca02c', 'Ω̄⁺'),
        ]:
            scores = p_all[cls_mask]
            feat   = feat_all[cls_mask]
            centers, means, errs = bin_mean_std(scores, feat, edges)
            in_range = (feat >= edges[0]) & (feat < edges[-1])
            global_mean = scores[in_range].numpy().mean() if in_range.any() else scores.numpy().mean()
            ax.errorbar(centers, means, yerr=errs, fmt='o-', capsize=3,
                        linewidth=1.5, color=color, label=label)
            ax.axhline(global_mean, color=color, ls='--', alpha=0.35, linewidth=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('Mean p(Anti)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Score vs kaon aggregate kinematics', fontsize=13)
    plt.tight_layout()
    out = 'plots/paper_kaon_agg_bias.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate paper figures for OmegaNet.')
    parser.add_argument('--rescore',  action='store_true',
                        help='Force re-scoring even if cache exists.')
    parser.add_argument('--permtest', action='store_true',
                        help='Run permutation feature importance via interpret_model.py.')
    args = parser.parse_args()

    os.makedirs('plots', exist_ok=True)

    p_all, y_all, raw_all, val_mask = load_or_score(rescore=args.rescore)

    print("\n── Figure 1: Score distribution ──────────────────────────────")
    plot_score_dist(p_all, y_all, val_mask)

    print("\n── Figure 2: Closure test ────────────────────────────────────")
    plot_closure_test(p_all, y_all, val_mask)

    print("\n── Figure 3: Purity vs cut + FOM ─────────────────────────────")
    lo_thresh, hi_thresh, bn_purity, pp_purity = plot_purity_vs_cut(p_all, y_all)
    print(f"  BN threshold: {lo_thresh:.4f}  (purity={bn_purity:.3f})")
    print(f"  PP threshold: {hi_thresh:.4f}  (purity={pp_purity:.3f})")

    print("\n── Figure 4: Bias check ──────────────────────────────────────")
    plot_bias_check(p_all, y_all, raw_all)

    print("\n── Figure 5: Ω pT subpopulation comparison ───────────────────")
    plot_omega_pt(p_all, y_all, raw_all, lo_thresh, hi_thresh, bn_purity, pp_purity)

    print("\n── Figure 6: Kaon aggregate kinematics bias ──────────────────")
    plot_kaon_aggregate_bias(p_all, y_all, raw_all)

    if args.permtest:
        print("\n── Figure 7: Permutation feature importance ──────────────────")
        from interpret_model import load_model, load_val_set, run_permutation_importance
        model   = load_model()
        val_set, _ = load_val_set()
        run_permutation_importance(model, val_set)

    print("\nDone. Outputs:")
    figs = ['paper_score_dist', 'paper_closure_test', 'paper_purity_vs_cut',
            'paper_bias_check', 'paper_omega_pt', 'paper_kaon_agg_bias']
    if args.permtest:
        figs.append('feature_importance')
    for f in figs:
        print(f"  plots/{f}.png")


if __name__ == '__main__':
    main()
