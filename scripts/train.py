import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import config
from pfn_model import OmegaPFN
from tqdm import tqdm


def run_training():
    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means']
    feature_stds = stats['stds']

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    labels = []

    print("Preparing Particle Flow Network Data (Unordered Sets)...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        target = y.squeeze().long()
        labels.append(target.item())

        # Node Features Scaling
        x[:, 1] = torch.clamp(x[:, 1], max=config.DPT_CLIP)
        x = (x - feature_means) / feature_stds

        # Simply store the set of Kaons and the label. No edge_index needed.
        dataset.append(Data(x=x, y=target))

    # Dynamic Weight Calculation
    labels_t = torch.tensor(labels)
    n_o, n_a = (labels_t == 0).sum().item(), (labels_t == 1).sum().item()
    weights = torch.tensor([(n_o + n_a) / (2 * n_o), (n_o + n_a) / (2 * n_a)]).to(config.DEVICE)
    print(f"Weights calculated -> Omega: {weights[0]:.2f}, Anti: {weights[1]:.2f}")

    # Splits
    import random
    random.seed(42)
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:split], batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset[split:], batch_size=config.BATCH_SIZE)

    # Initialize Deep Sets Model
    model = OmegaPFN(5, config.HIDDEN_CHANNELS, 2).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    best_score = -1.0

    print("Starting Training Loop...")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(config.DEVICE)
            optimizer.zero_grad()
            # PFN only needs the node features and the batch mapping
            out = model(batch.x, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

        model.eval()
        o_t, o_p, a_t, a_p = 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(config.DEVICE)
                out = model(batch.x, batch.batch)
                preds = out.argmax(dim=1)
                o_t += (batch.y == 0).sum().item()
                o_p += ((preds == 0) & (batch.y == 0)).sum().item()
                a_t += (batch.y == 1).sum().item()
                a_p += ((preds == 1) & (batch.y == 1)).sum().item()

        o_rec, a_rec = o_p / o_t if o_t > 0 else 0, a_p / a_t if a_t > 0 else 0
        score = o_rec + a_rec - 1.0
        scheduler.step(score)

        print(
            f"Epoch {epoch:02d} | Omega Rec: {o_rec:.3f} | Anti Rec: {a_rec:.3f} | Score: {score:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)


if __name__ == "__main__":
    run_training()