import torch
import os

# Paths and Naming
DATA_PATH = "data/balanced_omega_anti.pt"
STATS_PATH = "data/balanced_omega_anti_stats.pt"
MODEL_SAVE_PATH = "models/omega_pfn.pth"
os.makedirs("models", exist_ok=True)

# Architecture & Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
HIDDEN_CHANNELS = 64
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DPT_CLIP = 8.0