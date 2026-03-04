import torch
import os

# Paths and Naming
DATA_PATH = "data/balanced_omega_anti.pt"
MODEL_SAVE_PATH = "models/omega_pfn.pth"
os.makedirs("models", exist_ok=True)

# Architecture & Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
HIDDEN_CHANNELS = 64
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Physics Scaling & Standardization
# Order: [f_pt, d_pt, d_eta, d_phi, f_q]
FEATURE_MEANS = torch.tensor([0.604, 1.876, 0.009, 0.000, 0.000], dtype=torch.float)
FEATURE_STDS = torch.tensor([0.339, 0.724, 1.086, 1.815, 1.000], dtype=torch.float)
DPT_CLIP = 8.0