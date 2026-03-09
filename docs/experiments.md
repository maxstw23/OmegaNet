# Experiments Log

Tracks all training runs, their configurations, and outcomes.
Primary metric: **Score = Anti_recall + Omega_recall − 1** at argmax threshold (t = 0.5).
Secondary: sweep-optimum score (dense threshold scan).

---

## Baseline: BCE Transformer

| Run | Features | IN | Model | Score | Notes |
|-----|----------|----|-------|-------|-------|
| run5–15 | Various | 5–6 | OmegaTransformer | ~0.28–0.33 | Architecture search, loss weight tuning |
| run16 | f_pt, k_star, d_y, d_phi, cos_theta_star | 5 | OmegaTransformer | **0.3161** | Best BCE baseline |
| run18 | + EP features (o_cos_psi1, o_cos2_psi2, f_cos_psi1, f_cos2_psi2) | 9 | OmegaTransformer | 0.3124 | EP features added no signal; converged ep55, early stop ep80 |

---

## NCE / Density-Ratio Experiments

| Run | Features | Score | Notes |
|-----|----------|-------|-------|
| adv_run1–3 | f_pt, k_star, d_y, d_phi, cos_theta_star | — | Early NCE variants |
| adv_run4 | 5 features | 0.3057 | NCE sweep score; scale explosion (f_Anti std ≫ f_Omega std) |
| adv_run5 | 5 features | ~0.304 | NCE v2 with L2 reg + 4× Omega accum; f_anti std ≈ 7.8 still unstable |

---

## Feature Scan (train.py --features)

| Run | Feature Set | IN | Score | Notes |
|-----|-------------|----|-------|-------|
| run19 | f_pt, k_star, d_y, d_phi, cos_theta_star | 5 | **0.3152** | Matches run16 baseline; EP features confirmed inert |
| run20 | k_star, cos_theta_star | 2 | 0.1565 | Near-random — pair-rest-frame features alone carry almost no signal; f_pt, d_y, d_phi essential |
| run21 (running) | f_pt, k_star, d_y, d_phi, cos_theta_star | 5 | — | Restore best checkpoint (overwritten by run20) |

---

## GRL Adversarial (train_grl.py)

**Approach**: DANN-style gradient reversal for multiplicity debiasing.
- Main task: per-class-mean BCE (same as train.py)
- Adversary: predict log(n_kaons) from gradient-reversed CLS embedding
- Forces encoder to be invariant to kaon multiplicity (charge-balancing artifact)
- Alpha schedule: linear ramp 0 → 1.0 over 30 epochs

| Run | Features | IN | ADV_LAMBDA | Score | Notes |
|-----|----------|----|-----------|-------|-------|
| grl_run1 | 9 (with EP) | 9 | 0.1 | died @ep5 | CUDA OOM (two GPU jobs simultaneously) |
| grl_run2 | 9 (with EP) | 9 | 0.1 | 0.3078 | Below baseline; multiplicity debiasing adds no benefit; adv loss ~0.07 throughout (encoder reached GRL equilibrium early but didn't improve classifier) |

---

## Consistency Regularization (train_consistency.py)

**Approach**: Augmentation by random kaon dropout; require consistent scores.
- L_cons = MSE(p_orig, p_dropped) — augmented view must give same score
- drop_frac ramps 0.1 → 0.4 over 40 epochs
- Forces model to learn distributed multi-kaon patterns, not single-kaon shortcuts
- Uses same OmegaTransformer backbone as train.py

| Run | Features | IN | CONS_LAMBDA | Drop range | Score | Notes |
|-----|----------|----|------------|------------|-------|-------|
| cons_run1 | 9 (with EP) | 9 | 0.3 | 0.1→0.4 | ~0.312 | Converged to ~baseline; cons loss was already tiny by ep90 (0.017) — model naturally gives consistent scores; augmentation added no novel signal |

---

## Pseudo-Labeling EM (train_pseudolabel.py)

**Approach**: Iterative E/M steps for PU label denoising.
- E-step: score all Omega events with current model → weight w_i = (1 − p_anti)^γ
- M-step: train with weighted BCE — pair-produced Omega events (high p_anti) downweighted
- γ = 1.5 (sharper than linear), 4 EM iterations
- Iter 1 = standard BCE (uniform weights); subsequent iters use model-derived soft labels

| Run | Features | IN | γ | EM iters | Score | Notes |
|-----|----------|----|---|---------|-------|-------|
| pl_run1 | f_pt, k_star, d_y, d_phi, cos_theta_star (ran as 9-feat due to active config) | 9 | 1.5 | 4 | 0.3124 | Global best from iter1 (= standard BCE). Subsequent iters SHIFT the operating point but don't improve separability — see below |

**EM iteration breakdown for pl_run1:**

| Iter | Score @t=0.5 | Anti rec @t=0.5 | Omega rec @t=0.5 | Anti rec @t=0.55 | Omega weight mean |
|------|------------|-----------------|------------------|------------------|-----------------|
| 1 (BCE) | 0.3124 | 0.635 | 0.677 | 0.540 | 1.000 (uniform) |
| 2 | 0.3051 | 0.689 | 0.616 | 0.637 | 0.457 |
| 3 | 0.3021 | 0.703 | 0.599 | 0.662 | 0.457 |
| 4 | 0.3024 | 0.711 | 0.591 | 0.673 | 0.450 |

**Interpretation**: EM correctly downweights pair-produced Ω⁻ events (mean weight 0.45 ≈ expected fraction of pair-produced Ω⁻), shifting the model toward higher Anti recall. But the total AUC / sweep-optimal score does not improve — the BN-transport Ω⁻ events remain indistinguishable from pair-produced ones even with correct label noise treatment. The information ceiling is real.

---

## Event-Mixed Padding (run25)

**Motivation**: Global K⁻ pool sampling was confirmed to bias d_y_signed and o_y_abs —
fake K⁻ drawn without conditioning on the Omega's kinematics produced pathological
d_y_signed values that the Transformer exploited. Anti events with many padded kaons
scored systematically higher (mean padding fraction 0.465 in spike vs 0.285 in bulk).

**Fix**: Replace uniform global pool with event-mixed pool binned on (|y_Ω|, pT_Ω) quartiles
(4×4 = 16 bins), so fake kaons have realistic kinematic relationships to the Omega they are
paired with.

| Run | Features | IN | Score (argmax) | O@A=0.90 | Notes |
|-----|----------|----|----------------|----------|-------|
| run25 | f_pt, k_star, d_y, d_phi, cos_θ*, d_y_signed, o_y_abs | 7 | **0.3418** | **0.3345** | Event-mixed padding; clear improvement over prior ~0.31–0.32 ceiling |

**Result**: Argmax score 0.3418 (O=0.717, A=0.625) — first run to clearly exceed the ~0.32
ceiling seen across 21+ prior runs. The improvement confirms that the global pool was
introducing a real artifact that masked genuine physics signal.

---

## Summary of Score Ceiling

Across BCE/NCE/GRL/consistency/EM runs on the old globally-pooled balanced dataset, score
plateaued at ~0.30–0.32. After fixing the padding artifact with event-mixed sampling, the
effective ceiling rose to ~0.34.

**Key null results (old preprocessing):**
- Event-plane cosine features (EP alignment) provide no discriminating signal (run18 vs run16)
- NCE/density-ratio objective not better than BCE at this signal strength (adv_run4,5)
- GRL adversarial debiasing (multiplicity invariance) does not help — no residual multiplicity bias after preprocessing (grl_run2)
- Consistency regularization (kaon dropout augmentation) adds no novel signal — model already naturally consistent (cons_run1)
- EM pseudo-labeling correctly shifts the operating point (Anti recall 0.635 → 0.711) but does not improve sweep-optimal separability; confirms genuine physics ceiling, not a label-noise artifact
- Ultra-minimal feature set (k*, cos_θ*) collapses to near-random (0.157) — f_pt, d_y, d_phi are essential
- Score is robust to model size, loss weights, and feature count (runs 5–21)

**Physics conclusion**: The ~0.34 score (run25, event-mixed padding) represents the current
best estimate of achievable separability with these features. The padding fix was a real
data-quality improvement, not a model change.
