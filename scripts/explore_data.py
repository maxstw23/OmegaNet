"""
Data exploration: compare kaon feature distributions between Omega and Anti-Omega events.
Produces plots/data_exploration.png and prints summary statistics.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

OUT_PATH = "plots/data_exploration.png"
FEATURE_NAMES = ["f_pt", "k_star", "d_y", "d_phi", "o_pt", "cos_theta_star"]


def load_split_by_class(data_path):
    raw = torch.load(data_path)
    omega_kaons, anti_kaons = [], []
    omega_per_event, anti_per_event = [], []

    for entry in raw:
        x = entry['x'].numpy()   # (n_kaons, 6)
        y = entry['y'].item()    # 0=Omega, 1=Anti

        if y == 0:
            omega_kaons.append(x)
            omega_per_event.append(x)
        else:
            anti_kaons.append(x)
            anti_per_event.append(x)

    return (np.vstack(omega_kaons), np.vstack(anti_kaons),
            omega_per_event, anti_per_event)


def print_stats(label, arr):
    print(f"\n  {label} — {len(arr)} kaons")
    for i, name in enumerate(FEATURE_NAMES):
        col = arr[:, i]
        print(f"    {name:<8}: mean={col.mean():+.4f}  std={col.std():.4f}  "
              f"[{col.min():.3f}, {col.max():.3f}]")


def main():
    os.makedirs("plots", exist_ok=True)
    raw = torch.load(config.DATA_PATH)

    # ── per-event multiplicity ──────────────────────────────────────────────
    omega_mults, anti_mults = [], []
    omega_k, anti_k = [], []
    for entry in raw:
        x = entry['x'].numpy()
        y = entry['y'].item()
        if y == 0:
            omega_k.append(x)
            omega_mults.append(len(x))
        else:
            anti_k.append(x)
            anti_mults.append(len(x))

    omega_k = np.vstack(omega_k)
    anti_k  = np.vstack(anti_k)
    omega_mults = np.array(omega_mults)
    anti_mults  = np.array(anti_mults)

    n_omega_ev = len(omega_mults)
    n_anti_ev  = len(anti_mults)

    print("\n" + "=" * 60)
    print(f"Events : {n_omega_ev} Omega, {n_anti_ev} Anti  (ratio {n_omega_ev/n_anti_ev:.3f})")
    print(f"Kaons  : {len(omega_k)} Omega, {len(anti_k)} Anti")
    print(f"Kaons/event : Omega {omega_mults.mean():.2f}±{omega_mults.std():.2f}, "
          f"Anti {anti_mults.mean():.2f}±{anti_mults.std():.2f}")

    print_stats("Omega kaons", omega_k)
    print_stats("Anti  kaons", anti_k)

    # All kaons are opposite-sign (strangeness partners); no same/opp split needed.

    # ── ΔR = sqrt(d_eta^2 + d_phi^2) ────────────────────────────────────────
    def dr(arr): return np.sqrt(arr[:, 2]**2 + arr[:, 3]**2)
    omega_dr = dr(omega_k)
    anti_dr  = dr(anti_k)
    print(f"\n  ΔR — Omega: mean={omega_dr.mean():.4f} std={omega_dr.std():.4f}")
    print(f"  ΔR — Anti : mean={anti_dr.mean():.4f} std={anti_dr.std():.4f}")

    # ── per-event aggregate features ─────────────────────────────────────────
    def per_event_agg(events):
        mean_dr, min_dr, mean_fpt, mean_dpt = [], [], [], []
        for x in events:
            drs = np.sqrt(x[:, 2]**2 + x[:, 3]**2)
            mean_dr.append(drs.mean())
            min_dr.append(drs.min())
            mean_fpt.append(x[:, 0].mean())
            mean_dpt.append(x[:, 1].mean())
        return (np.array(mean_dr), np.array(min_dr),
                np.array(mean_fpt), np.array(mean_dpt))

    o_mdr, o_mindr, o_mfpt, o_mdpt = per_event_agg(
        [entry['x'].numpy() for entry in raw if entry['y'].item() == 0])
    a_mdr, a_mindr, a_mfpt, a_mdpt = per_event_agg(
        [entry['x'].numpy() for entry in raw if entry['y'].item() == 1])

    print(f"\n  Per-event mean ΔR  — Omega: {o_mdr.mean():.4f}, Anti: {a_mdr.mean():.4f}")
    print(f"  Per-event min  ΔR  — Omega: {o_mindr.mean():.4f}, Anti: {a_mindr.mean():.4f}")
    print(f"  Per-event mean f_pt— Omega: {o_mfpt.mean():.4f}, Anti: {a_mfpt.mean():.4f}")
    print("=" * 60)

    # ── Plotting ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Data Exploration: Omega vs Anti-Omega Kaon Features", fontsize=13)

    colors = {'omega': 'steelblue', 'anti': 'tomato'}
    alpha = 0.55

    # Row 1: raw feature distributions (5 features)
    feat_ranges = {
        'f_pt':   (0, 3.5, 60),
        'k_star': (0, 4.0, 60),
        'd_eta':  (-3, 3, 60),
        'd_phi':  (-np.pi, np.pi, 60),
        'o_pt':   (0, 3.5, 60),
    }
    for i, (name, (lo, hi, nb)) in enumerate(feat_ranges.items()):
        ax = fig.add_subplot(4, 5, i + 1)
        bins = np.linspace(lo, hi, nb)
        ax.hist(omega_k[:, i], bins=bins, density=True, alpha=alpha,
                color=colors['omega'], label='Omega')
        ax.hist(anti_k[:, i],  bins=bins, density=True, alpha=alpha,
                color=colors['anti'],  label='Anti')
        ax.set_title(name, fontsize=10)
        ax.set_ylabel('Density')
        if i == 0: ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 2: ΔR distributions
    ax = fig.add_subplot(4, 5, 6)
    bins = np.linspace(0, 6, 60)
    ax.hist(omega_dr, bins=bins, density=True, alpha=alpha, color=colors['omega'], label='Omega')
    ax.hist(anti_dr,  bins=bins, density=True, alpha=alpha, color=colors['anti'],  label='Anti')
    ax.set_title('ΔR = √(Δη²+Δφ²)', fontsize=10)
    ax.set_xlabel('ΔR')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 3: per-event aggregates
    for idx, (o_arr, a_arr, title, xlab) in enumerate([
        (o_mdr,   a_mdr,   'Per-event mean ΔR', 'mean ΔR'),
        (o_mindr, a_mindr, 'Per-event min ΔR',  'min ΔR'),
        (o_mfpt,  a_mfpt,  'Per-event mean f_pt','mean f_pt [GeV/c]'),
        (o_mdpt,  a_mdpt,  'Per-event mean d_pt','mean d_pt [GeV/c]'),
        (omega_mults.astype(float), anti_mults.astype(float), 'Kaon multiplicity', 'N kaons'),
    ]):
        ax = fig.add_subplot(4, 5, 11 + idx)
        lo = min(o_arr.min(), a_arr.min())
        hi = max(o_arr.max(), a_arr.max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(o_arr, bins=bins, density=True, alpha=alpha, color=colors['omega'], label='Omega')
        ax.hist(a_arr, bins=bins, density=True, alpha=alpha, color=colors['anti'],  label='Anti')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlab, fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
