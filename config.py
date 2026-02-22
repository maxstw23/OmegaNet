import torch

# Data Paths
DATA_PATH = "data/balanced_omega_anti.pt"
MODEL_SAVE_PATH = "physics_gat_model.pth"

# Model Hyperparameters
HIDDEN_CHANNELS = 64
ATTENTION_HEADS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4

# Training Settings
BATCH_SIZE = 128
TRAIN_SPLIT = 0.8
MAX_EPOCHS = 100
PATIENCE = 10
RANDOM_SEED = 42

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0
PIN_MEMORY = False