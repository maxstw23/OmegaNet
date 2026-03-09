"""train_consistency.py — Consistency regularization via kaon dropout augmentation.

Idea: for each training event, randomly drop DROP_FRAC of kaons to create an
augmented view.  A good model should assign similar scores to both views,
since the production mechanism is a property of the event, not of any specific
kaon.  The consistency loss penalises score disagreement between views.

This forces the model to learn features robust to kaon subsampling — i.e., it
must capture distributed multi-kaon patterns rather than over-relying on any
single kaon.  Augmentation is in-memory (no preprocessing change needed).

Loss:
    L = L_cls + λ_cons * L_cons
    L_cls  = per-class-mean BCE on original events (as in train.py)
    L_cons = mean squared difference of p(x) vs p(x_dropped)

Kaon dropout schedule: DROP_FRAC starts at DROP_START, linearly increases to
DROP_MAX over DROPOUT_WARMUP_EPOCHS, then holds.  Gradual increase avoids
pathological degenerate events at the start of training.

Checkpoint: argmax score (Anti_recall + Omega_recall − 1), same as train.py.
"""
import sys
import os
import copy
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import config
from transformer_model import OmegaTransformer
from tqdm import tqdm

# ── Consistency hyperparameters ───────────────────────────────────────────────
DROP_START          = 0.1   # initial kaon drop fraction
DROP_MAX            = 0.4   # max kaon drop fraction (reached at DROPOUT_WARMUP_EPOCHS)
DROPOUT_WARMUP_EPOCHS = 40  # epochs to ramp DROP_FRAC from DROP_START → DROP_MAX
CONS_LAMBDA         = 0.3   # consistency loss weight
MIN_KAONS           = 2     # minimum kaons to retain after dropping
CHECKPOINT_THRESHOLD = 0.55
EMA_DECAY           = 0.999
CONS_SAVE_PATH      = "models/omega_consistency.pth"


def get_next_run_number():
    os.makedirs("logs", exist_ok=True)
    existing = glob.glob("logs/cons_run*.log")
    nums = [
        int(os.path.basename(f).replace("cons_run", "").replace(".log", ""))
        for f in existing
        if os.path.basename(f).replace("cons_run", "").replace(".log", "").isdigit()
    ]
    return max(nums) + 1 if nums else 1


class KaonDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    n_feat  = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, n_feat)
    mask   = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.shape[0]
        padded[i, :n] = x
        mask[i, :n] = False
    return padded, torch.stack(ys), mask


def kaon_dropout(x_batch, mask_batch, drop_frac, min_kaons=2):
    """Randomly drop drop_frac of real kaons from each event in the batch.

    Args:
        x_batch:    (B, T, F) padded feature tensor
        mask_batch: (B, T) bool mask (True = padding)
        drop_frac:  fraction of real kaons to drop (randomly per event)
        min_kaons:  minimum real kaons to keep

    Returns:
        x_dropped:    (B, T, F) new padded tensor with fewer real kaons
        mask_dropped: (B, T) updated padding mask
    """
    B, T, F = x_batch.shape
    x_dropped   = torch.zeros_like(x_batch)
    mask_dropped = torch.ones(B, T, dtype=torch.bool, device=x_batch.device)

    for i in range(B):
        real_idx = (~mask_batch[i]).nonzero(as_tuple=True)[0]  # indices of real kaons
        n_real   = len(real_idx)
        n_drop   = max(0, int(n_real * drop_frac))
        n_keep   = max(min_kaons, n_real - n_drop)
        if n_keep >= n_real:
            # Nothing to drop — copy as is
            x_dropped[i, :n_real]   = x_batch[i, :n_real]
            mask_dropped[i, :n_real] = False
        else:
            keep_idx = real_idx[torch.randperm(n_real)[:n_keep]]
            keep_idx = keep_idx.sort().values
            x_dropped[i, :n_keep]   = x_batch[i, keep_idx]
            mask_dropped[i, :n_keep] = False

    return x_dropped, mask_dropped


def run_training():
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    run_number = get_next_run_number()
    log_path = f"logs/cons_run{run_number}.log"

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    stats = torch.load(config.STATS_PATH)
    feature_means = stats['means'][config.FEATURE_IDX]
    feature_stds  = stats['stds'][config.FEATURE_IDX]

    raw_data = torch.load(config.DATA_PATH)
    dataset, labels = [], []

    print("Preparing consistency dataset...")
    for entry in tqdm(raw_data):
        x, y = entry['x'], entry['y']
        target = y.squeeze().long()
        labels.append(target.item())
        x = x[:, config.FEATURE_IDX]
        if config.KSTAR_IDX is not None:
            x[:, config.KSTAR_IDX] = torch.clamp(x[:, config.KSTAR_IDX], max=config.KSTAR_CLIP)
        x = (x - feature_means) / feature_stds
        dataset.append((x, target))

    labels_t = torch.tensor(labels)
    n_o = (labels_t == 0).sum().item()
    n_a = (labels_t == 1).sum().item()

    log(f"Consistency Run {run_number} | features={config.FEATURE_NAMES} | IN_CHANNELS={config.IN_CHANNELS}")
    log(f"Dataset: {n_o} Omega, {n_a} Anti-Omega")
    log(f"DROP_START={DROP_START} → DROP_MAX={DROP_MAX} over {DROPOUT_WARMUP_EPOCHS} epochs | CONS_LAMBDA={CONS_LAMBDA}")

    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_loader = DataLoader(
        KaonDataset(dataset[:split]), batch_size=config.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        KaonDataset(dataset[split:]), batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    model = OmegaTransformer(
        in_channels=config.IN_CHANNELS,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT_RATE,
    ).to(config.DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"OmegaTransformer: {n_params:,} parameters")

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_score = -1.0
    best_anti_rec = -1.0
    best_a_rec_055 = -1.0
    epochs_no_improvement = 0

    log("Starting consistency training...\n")
    for epoch in range(1, config.EPOCHS + 1):
        drop_frac = DROP_START + (DROP_MAX - DROP_START) * min(1.0, epoch / DROPOUT_WARMUP_EPOCHS)

        model.train()
        epoch_cls, epoch_cons, n_steps = 0., 0., 0
        for x, y, mask in train_loader:
            x, y, mask = x.to(config.DEVICE), y.to(config.DEVICE), mask.to(config.DEVICE)

            optimizer.zero_grad()

            # Original forward pass
            out_orig = model(x, mask)
            p_orig = torch.softmax(out_orig, dim=1)[:, 1].clamp(1e-7, 1 - 1e-7)

            is_a = (y == 1)
            is_o = (y == 0)
            loss_cls = -torch.log(p_orig[is_a]).mean() + -torch.log(1 - p_orig[is_o]).mean()

            # Augmented view: kaon dropout
            x_drop, mask_drop = kaon_dropout(x, mask, drop_frac, MIN_KAONS)
            out_drop = model(x_drop, mask_drop)
            p_drop = torch.softmax(out_drop, dim=1)[:, 1].clamp(1e-7, 1 - 1e-7)

            # Consistency: p_orig and p_drop should agree
            loss_cons = F.mse_loss(p_orig, p_drop.detach())

            loss = loss_cls + CONS_LAMBDA * loss_cons
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(EMA_DECAY).add_(live_p.data, alpha=1.0 - EMA_DECAY)

            epoch_cls  += loss_cls.item()
            epoch_cons += loss_cons.item()
            n_steps += 1

        ema_model.eval()
        all_p_anti, all_y = [], []
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, mask = x.to(config.DEVICE), mask.to(config.DEVICE)
                out = ema_model(x, mask)
                p_anti = torch.softmax(out, dim=1)[:, 1]
                all_p_anti.append(p_anti.cpu())
                all_y.append(y)

        p_anti     = torch.cat(all_p_anti)
        labels_val = torch.cat(all_y)
        is_omega   = (labels_val == 0)
        is_anti    = (labels_val == 1)

        preds_argmax = (p_anti >= 0.5).long()
        o_rec = ((preds_argmax == 0) & is_omega).sum().item() / is_omega.sum().item()
        a_rec = ((preds_argmax == 1) & is_anti).sum().item()  / is_anti.sum().item()
        score = o_rec + a_rec - 1.0

        preds_055  = (p_anti >= CHECKPOINT_THRESHOLD).long()
        a_rec_055  = ((preds_055 == 1) & is_anti).sum().item() / is_anti.sum().item()

        scheduler.step(score)

        line = (
            f"Epoch {epoch:03d} | drop={drop_frac:.2f} | "
            f"cls={epoch_cls/n_steps:.4f} cons={epoch_cons/n_steps:.4f} | "
            f"Omega Rec: {o_rec:.3f} | Anti Rec: {a_rec:.3f} | "
            f"A@{CHECKPOINT_THRESHOLD:.2f}: {a_rec_055:.3f} | Score: {score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"No-improve: {epochs_no_improvement}/{config.EARLY_STOP_PATIENCE}"
        )
        log(line)

        if score > best_score:
            best_score    = score
            best_anti_rec = a_rec
            best_a_rec_055 = a_rec_055
            epochs_no_improvement = 0
            torch.save(ema_model.state_dict(), CONS_SAVE_PATH)
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= config.EARLY_STOP_PATIENCE:
            log(f"\nEarly stopping: no improvement for {config.EARLY_STOP_PATIENCE} epochs.")
            break

    summary = (
        f"\nBest Score: {best_score:.4f} | Anti (argmax): {best_anti_rec:.4f} | "
        f"A@{CHECKPOINT_THRESHOLD:.2f}: {best_a_rec_055:.4f}  →  saved to {CONS_SAVE_PATH}"
    )
    log(summary)
    log(f"Log saved to {log_path}")


if __name__ == "__main__":
    run_training()
