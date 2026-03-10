# Experiments Log

Primary metric: **O@A=0.90** (Omega recall at fixed Anti recall = 0.90).
All other scores (argmax, sweep-optimum) are internal training diagnostics only.

---

## BCE Transformer Baseline (old global-pool preprocessing)

| Run | Features | O@A=0.90 | Notes |
|-----|----------|----------|-------|
| run5–15 | Various 5–6 feat | — | Architecture search; scores recorded as argmax only |
| run16 | f_pt, k_star, d_y, d_phi, cos_θ* | ~0.26 | Best old-preprocessing BCE baseline |
| run18 | + EP cosines (×4) | — | EP features add no signal |
| run19 | f_pt, k_star, d_y, d_phi, cos_θ* | — | Confirms run16; EP features inert |
| run20 | k_star, cos_θ* only | — | Near-random — f_pt, d_y, d_phi are essential |

---

## Alternative Objectives (old preprocessing)

All run on same dataset with global K⁻ pool padding.

| Method | O@A=0.90 | Notes |
|--------|----------|-------|
| NCE/density-ratio (adv_run4–5) | — | Unstable (scale explosion); no improvement over BCE |
| GRL adversarial (grl_run2) | — | Multiplicity debiasing adds no benefit |
| Consistency regularization (cons_run1) | — | Augmentation adds no signal; model already consistent |
| EM pseudo-labeling (pl_run1) | — | Shifts operating point but does not improve separability |

Note: O@A=0.90 was not tracked for these runs; scores above were recorded as argmax at t=0.5.

---

## Event-Mixed Padding (run25)

**Problem**: Global K⁻ pool sampling biased d_y_signed — fake K⁻ had |y_K| values independent
of the specific Omega's |y_Ω|, producing pathological rapidity gaps. Anti events with many
padded kaons scored systematically higher (padding fraction 0.465 in score-spike vs 0.285 in bulk).

**Fix**: Event-mixed K⁻ pool binned on (|y_Ω|, pT_Ω) quartiles (4×4 = 16 bins).

| Run | Features | O@A=0.90 | Notes |
|-----|----------|----------|-------|
| run25 | f_pt, k_star, d_y, d_phi, cos_θ*, d_y_signed, o_y_abs | **0.3345** | Event-mixed padding; first clear improvement over old ceiling |

Note: run25 included `o_y_abs` which was subsequently removed (introduces biased rapidity shift).

**Run 26 interpretation (interpret_model.py):**

Permutation importance (score drop, 15 repeats):

| Feature | Drop | Relative |
|---|---|---|
| d_y_signed | 0.221 | 66% |
| d_y | 0.200 | 60% |
| f_pt | 0.134 | 40% |
| k_star | 0.077 | 23% |
| cos_theta_star | 0.076 | 23% |
| d_phi | 0.019 | 6% |

The rapidity features (d_y_signed and d_y) dominate — together accounting for ~60–66% of the
signal. k* and cos_theta_star contribute equally (~23% each). d_phi carries almost no signal.
No feature has negative importance (no artifact dependency detected).

Attention analysis: attention-weighted feature differences between Omega and Anti are small
(largest Δ: d_y_signed +0.049, d_y +0.029), consistent with the weak per-kaon signal — the
model exploits multi-kaon patterns rather than individual distinctive kaons.

---

## Key Null Results

- EP cosine features (event-plane alignment) add no signal
- NCE/density-ratio no better than BCE at this signal strength
- GRL adversarial debiasing (multiplicity invariance) does not help
- Consistency regularization (kaon dropout) adds no novel signal
- EM pseudo-labeling shifts operating point but not separability — confirms physics ceiling, not label-noise artifact
- Ultra-minimal [k*, cos_θ*] collapses to near-random — f_pt, d_y, d_phi essential
- Global K⁻ pool padding was a real artifact masking genuine signal
