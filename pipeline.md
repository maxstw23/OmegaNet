# Pipeline

## Overview

```
data/omega_kaon_all.root
        │
        ▼
scripts/preprocess_data.py  →  data/balanced_omega_anti.pt
        │
        ▼
scripts/train.py  →  models/<some model>.pth
        │
        ▼
scripts/evaluate_physics.py  →  recall metrics / physics interpretation
```

A helper script `scripts/inspect_data.py` can be run after preprocessing to verify
feature statistics and catch any data issues before training.

---

## Step 1: Preprocessing

```
python scripts/preprocess_data.py
```

Reads `data/omega_kaon_all.root` (TTree: `ml_tree`) and outputs
`data/balanced_omega_anti.pt` — a list of per-event dicts containing node features
and a binary label (0 = Ω⁻, 1 = Ω̄⁺).

Each event is processed as follows:
1. Kaon multiplicities are balanced so that N(same-sign) = N(opposite-sign) relative
   to the Omega, by sampling from a global kaon momentum pool built across all events.
2. Raw momenta are converted to cylindrical coordinates.
3. Per-kaon node features are computed relative to the Omega:
   `[f_pt, d_pt, d_eta, d_phi, f_q]`, where `f_q` is the kaon charge sign relative
   to the Omega (same-sign = +1, opposite-sign = −1).

---

## Step 2: Inspection (optional)

```
python scripts/inspect_data.py
```

Prints feature statistics (mean, std, min, max) and runs sanity checks on the
preprocessed dataset.

---

## Step 3: Training

```
python scripts/train.py
```

Trains the model on `data/balanced_omega_anti.pt` and saves the best checkpoint. Progress is printed per epoch showing Omega recall,
Anti-Omega recall, and the physics score (Omega_recall + Anti_recall − 1).

---

## Step 4: Evaluation

```
python scripts/evaluate_physics.py
```

Loads the saved model and reports the physics metrics on the validation set:
- **Omega Recall**: ideally ≈ 0.5 (the BN-carrying half)
- **Anti-Omega Recall**: ideally ≈ 1.0
- **Physics Score**: Omega_recall + Anti_recall − 1
