# CLAUDE.md

This project uses ML to study Omega hyperon production in heavy-ion collisions at RHIC BES-II energies.

## Key References

- **`physics.md`** — Physics motivation: baryon number transport, gluon junction picture, and why Omega is the probe of choice.
- **`method.md`** — ML strategy: how the Omega/anti-Omega classification task encodes the physics goal, and how charge blinding is implemented.
- **`pipeline.md`** — How to run the project end-to-end: preprocessing, inspection, training, and evaluation.

## Environment

- Python venv at `venv/`. Always use `venv/bin/python` to run scripts.
- Scripts must be run from the project root so that relative paths in `config.py` resolve correctly.
