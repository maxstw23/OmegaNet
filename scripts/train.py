import torch
import torch.nn.functional as F
import numpy as np
import time
import gc
from datetime import datetime
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from gat_model import PhysicsGAT


class EarlyStopping:
    def __init__(self, patience=config.PATIENCE, path=config.MODEL_SAVE_PATH):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def run_training():
    start_time = time.time()
    print(f"--- Restarting Session: {datetime.now().strftime('%H:%M:%S')} ---")

    # 1. Load raw tensors
    raw_data = torch.load(config.DATA_PATH)

    # 2. Extract Weights
    labels = np.array([entry['y'].item() for entry in raw_data])
    n_omega = np.sum(labels == 0)
    n_anti = np.sum(labels == 1)
    w_omega = len(labels) / (2.0 * n_omega)
    w_anti = len(labels) / (2.0 * n_anti)
    class_weights = torch.tensor([w_omega, w_anti], dtype=torch.float).to(config.DEVICE)

    # 3. Build Dataset
    dataset = []
    print("Building Graphs...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        n_nodes = x.size(0)

        if n_nodes > 1:
            edge_index = torch.combinations(torch.arange(n_nodes), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)

        dataset.append(Data(x=x, edge_index=edge_index, y=y.squeeze()))

    del raw_data
    gc.collect()

    # 4. Split
    train_data, val_data = train_test_split(dataset, test_size=(1 - config.TRAIN_SPLIT),
                                            random_state=config.RANDOM_SEED)

    # 5. Loaders
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)

    model = PhysicsGAT(config.HIDDEN_CHANNELS, config.ATTENTION_HEADS).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    stopper = EarlyStopping()

    print(f"Starting Training Loop. VRAM Allocated: {torch.cuda.memory_allocated(config.DEVICE) / 1e9:.2f}GB")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        # Using a fresh tqdm bar for every epoch to monitor internal progress
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for batch in pbar:
            batch = batch.to(config.DEVICE)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(config.DEVICE)
                out = model(batch.x, batch.edge_index, batch.batch)
                val_loss += criterion(out, batch.y).item() * batch.num_graphs
                correct += int((out.argmax(dim=1) == batch.y).sum())

        avg_val_loss = val_loss / len(val_data)

        # Periodic update
        elapsed = time.time() - start_time
        print(
            f"\n>> Epoch {epoch:03d} Summary | Val Loss: {avg_val_loss:.4f} | Acc: {correct / len(val_data):.4f} | Time: {int(elapsed // 60)}m")

        # Explicit memory release after validation
        torch.cuda.empty_cache()
        gc.collect()

        stopper(avg_val_loss, model)
        if stopper.early_stop:
            print("Patience reached. Best model saved.")
            break

    print(f"\n--- Training Finished ---")


if __name__ == "__main__":
    run_training()