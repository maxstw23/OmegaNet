import torch
import os

# Paths and Naming
DATA_PATH = "data/balanced_omega_anti.pt"
STATS_PATH = "data/balanced_omega_anti_stats.pt"
MODEL_SAVE_PATH = "models/omega_transformer.pth"
os.makedirs("models", exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
EPOCHS = 100
EARLY_STOP_PATIENCE = 25  # stop if no new best Anti recall for this many epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PFN hyperparameters (kept for evaluate_physics.py backward compat)
HIDDEN_CHANNELS = 64

KSTAR_CLIP = 8.0

# Transformer Hyperparameters
IN_CHANNELS = 6        # [f_pt, k_star, d_y, d_phi, o_pt, cos_theta_star]
D_MODEL = 128
NHEAD = 4              # head_dim = 32
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT_RATE = 0.1