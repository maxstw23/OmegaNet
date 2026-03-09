# Lessons — Hard-Won Project Rules

## The Only Metric That Matters: O@A=0.90

**The sweep-optimum score (Anti_recall + Omega_recall − 1 maximised over all thresholds)
is IRRELEVANT. Never report it, never lead with it, never use it to compare runs.**

The operating point is fixed: **Anti recall = 0.90**.
The metric is: **Omega recall at that fixed Anti recall** (O@A=0.90).

This has been stated multiple times. Forgetting it wastes the user's time.

Corollary: the "argmax score" at t=0.5 is also not the primary metric — it is a training
checkpoint criterion only, not a result to report.

When summarising a training run, report **O@A=0.90 first and only**.
