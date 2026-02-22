import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import config
from gat_model import PhysicsGAT


def evaluate_model():
    # 1. Load and Rebuild Graph Objects for validation
    raw_data = torch.load(config.DATA_PATH)
    dataset = []
    for entry in raw_data:
        n_nodes = entry['x'].size(0)
        edge_index = torch.combinations(torch.arange(n_nodes), r=2).t() if n_nodes > 1 else torch.tensor([[0], [0]],
                                                                                                         dtype=torch.long)
        dataset.append(Data(x=entry['x'], edge_index=edge_index, y=entry['y'].squeeze()))

    _, val_data = train_test_split(dataset, test_size=(1 - config.TRAIN_SPLIT), random_state=config.RANDOM_SEED)
    loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Load Model
    model = PhysicsGAT(config.HIDDEN_CHANNELS, config.ATTENTION_HEADS).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            all_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    # 3. Generate Visuals
    ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, display_labels=['Omega', 'Anti-Omega'],
                                            cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Physics Signal Detection')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")

    print(f"Evaluation complete. AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    evaluate_model()