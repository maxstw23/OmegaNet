import torch
from torch_geometric.data import Data, DataLoader
import config
from pfn_model import OmegaPFN
from tqdm import tqdm


def evaluate():
    device = config.DEVICE
    print(f"Loading Deep Sets Model: {config.MODEL_SAVE_PATH}")

    model = OmegaPFN(5, config.HIDDEN_CHANNELS, 2).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means']
    feature_stds = stats['stds']

    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    print("Preparing Evaluation Sets...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']

        # Consistent Scaling
        x[:, 1] = torch.clamp(x[:, 1], max=config.DPT_CLIP)
        x = (x - feature_means) / feature_stds

        dataset.append(Data(x=x, y=y.squeeze().long()))

    # Exact same 20% validation split (must match shuffle in train.py)
    import random
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(0.8 * len(dataset))
    val_loader = DataLoader(dataset[split_idx:], batch_size=config.BATCH_SIZE)

    o_t, o_p, a_t, a_p = 0, 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.batch)
            preds = out.argmax(dim=1)

            o_t += (batch.y == 0).sum().item()
            o_p += ((preds == 0) & (batch.y == 0)).sum().item()
            a_t += (batch.y == 1).sum().item()
            a_p += ((preds == 1) & (batch.y == 1)).sum().item()

    o_rec = o_p / o_t if o_t > 0 else 0
    a_rec = a_p / a_t if a_t > 0 else 0
    score = o_rec + a_rec - 1.0

    print("\n" + "=" * 40)
    print(f"  Omega Recall (Junction?): {o_rec:.4f}")
    print(f"   Anti Recall:             {a_rec:.4f}")
    print(f" Physics Score:             {score:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate()