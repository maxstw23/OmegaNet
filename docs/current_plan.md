# Active Plan — Interpretation Analysis (run26)

## Goal

Run the attention and permutation importance analyses on the run26 checkpoint
(6 features: f_pt, k*, d_y, d_phi, cos_θ*, d_y_signed) and document findings.

## Script

`scripts/interpret_model.py` — already implements both analyses, no changes needed.
Uses current `config.FEATURE_NAMES` (6 features) and `models/omega_transformer.pth` (run26).

## Analysis 1: Attention

For each validation event, extract the CLS→kaon attention weights from the final
Transformer layer. Compute:
- **Attention-weighted mean** of each feature per event: the "effective feature
  value" the model acts on
- **Most-attended kaon** features per event: what single kaon the model focuses on

Compare Omega vs Anti distributions. Key questions:
- Does the model attend more to low-k* kaons (physically motivated: femtoscopic
  correlation)?
- Does the attention-weighted k* distribution differ between classes while the raw
  marginal does not? (Would confirm the model exploits within-event structure.)
- Is d_y_signed distinctive for the most-attended kaon in Omega events?

Outputs: `plots/attention_analysis.png`

## Analysis 2: Feature Permutation Importance

Permute each feature globally across all validation events (15 repeats), measure
score drop. The internal metric is argmax score (Anti+Omega recall − 1) — acceptable
here as a relative ranking metric, not for absolute comparison across runs.

Key questions:
- Which features drive the classification? k* and d_y expected to be most important.
- Is d_y_signed (the new signed feature replacing absolute d_y in the Omega index)
  informative on its own?
- Do any features show negative importance (permuting improves score), indicating the
  model may be exploiting a residual artifact?

Outputs: `plots/feature_importance.png`

## Execution

```bash
venv/bin/python scripts/interpret_model.py
```

Estimated runtime: ~5 min (attention analysis fast; permutation needs 6 × 15 = 90
model evaluations on the val set).

## Post-Analysis

Update `docs/interpretation.md` with:
- Actual feature importance rankings from run26
- Whether attention pattern is consistent with physics (low-k* focus)
- Any unexpected findings (negative importance, unexpected top feature)

Update `docs/experiments.md` run26 entry with interpretation summary.
