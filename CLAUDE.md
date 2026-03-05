# CLAUDE.md

This project uses ML to study Omega hyperon production in heavy-ion collisions at RHIC BES-II energies.

## Key References

- **`physics.md`** — Physics motivation: baryon number transport, gluon junction picture, and why Omega is the probe of choice.
- **`method.md`** — ML strategy: how the Omega/anti-Omega classification task encodes the physics goal, and how charge blinding is implemented.
- **`pipeline.md`** — How to run the project end-to-end: preprocessing, inspection, training, and evaluation.
- **`data_exploration.md`** — Findings from exploratory analysis of the preprocessed dataset: feature distributions, ΔR, k*, and signal-to-noise discussion.
- **`interpretation.md`** — Plan for four model interpretability analyses (attention, permutation importance, gradient attribution, CLS embedding correlation).

## Environment

- Python venv at `venv/`. Always use `venv/bin/python` to run scripts.
- Scripts must be run from the project root so that relative paths in `config.py` resolve correctly.
- CUDA: RTX 3070, CUDA 12.8, PyTorch 2.6.0+cu124.

## Model

Current model: **OmegaTransformer** (`models/transformer_model.py`)
- 2-layer Pre-LN Transformer, CLS token, no positional encoding, 275K parameters
- Checkpoint: `models/omega_transformer.pth`

Config (`config.py`):
- `IN_CHANNELS = 6` — feature dimension per kaon
- `D_MODEL = 128`, `NHEAD = 4`, `NUM_LAYERS = 2`, `DIM_FEEDFORWARD = 256`
- `KSTAR_CLIP = 8.0` — applied to k* (feature index 1) before normalisation

## Features

Six features per kaon, computed in `scripts/preprocess_data.py`:

| Index | Name | Description |
|---|---|---|
| 0 | `f_pt` | Kaon transverse momentum [GeV/c], capped at 2 GeV/c |
| 1 | `k_star` | Lorentz-invariant relative momentum in kaon-Ω pair rest frame [GeV/c] |
| 2 | `d_y` | y_kaon − y_Omega (true rapidity difference, boost-invariant, uses PDG masses) |
| 3 | `d_phi` | φ_kaon − φ_Omega, wrapped to [−π, π] |
| 4 | `o_pt` | Omega transverse momentum [GeV/c] — broadcast (same for all kaons in event) |
| 5 | `cos_theta_star` | cos(θ*) = k*_z / \|k*\| — beam-axis angle in pair rest frame (source elongation probe) |

k* uses PDG masses: m_K = 0.493677 GeV/c², m_Ω = 1.67245 GeV/c².
Feature means/stds are auto-computed by preprocessing and saved to `data/balanced_omega_anti_stats.pt`.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/preprocess_data.py` | ROOT → `data/balanced_omega_anti.pt` + stats |
| `scripts/inspect_data.py` | Print feature statistics and sanity checks |
| `scripts/explore_data.py` | Full data exploration plots → `plots/data_exploration.png` |
| `scripts/train.py` | Train OmegaTransformer, save best checkpoint |
| `scripts/evaluate_physics.py` | Threshold scan + physics metrics on val set |
| `scripts/plot_recall_tradeoff.py` | Dense threshold sweep + recall tradeoff curve → `plots/recall_tradeoff.png` |
| `scripts/interpret_model.py` | Attention analysis + feature permutation importance → `plots/` |

## Training

Loss: **per-class-mean asymmetric BCE** (replaces weighted CrossEntropyLoss):
```
loss = anti_weight × mean(−log p_Anti  | Anti events)
     +               mean(−log p_Omega | Omega events)
```
- Class imbalance handled automatically (each event weighted equally within its class)
- `anti_weight = 1` → balanced symmetric solution; `> 1` → emphasises Anti recall
- Checkpoint saved by best argmax score (Anti recall + Omega recall − 1 at t = 0.5)
- Scheduler: ReduceLROnPlateau on argmax score, patience = 5, factor = 0.5

Key training findings:
- Score plateaus at ~0.40 regardless of loss weights, model size, or features
- Per-kaon feature distributions are nearly identical between Omega and Anti events
- Signal lives in multi-kaon correlations — visible as clear separation in P(Anti) score distributions
- Threshold is a post-hoc choice; operating point for physics analysis set by val-set threshold sweep

## Physics Goal

Train a classifier on Ω vs Ω̄ events using kaon kinematics only (charge-blinded).
Expected outcome if BN-carrying signal exists:
- **Anti recall → 1.0** (all Ω̄⁺ are pair-produced, distinguishable)
- **Omega recall → 0.5** (only BN-carrying Ω⁻ classified correctly; pair-produced look like Ω̄⁺)
- The recall asymmetry at the chosen operating point estimates the BN-carrying fraction f_BN
