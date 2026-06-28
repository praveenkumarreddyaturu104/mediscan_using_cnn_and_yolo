"""
cnn_classifier.py
-----------------
Chest X-ray disease classifier using DenseNet121.
Dataset : NIH Chest X-Ray14 (112,120 images, 14 disease labels)
Download: kaggle datasets download -d nih-chest-xrays/data
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from PIL import Image
import pandas as pd
import numpy as np

# ── 14 disease labels from NIH dataset ────────────────────────────
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# ── image transforms (ImageNet mean/std since we fine-tune) ───────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ── dataset ───────────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """
    Reads NIH Data_Entry_2017.csv.
    Each image can have multiple labels separated by '|'.
    We encode them as a multi-hot vector of length 14.
    """
    def __init__(self, csv_path, img_dir, transform=None):
        self.df        = pd.read_csv(csv_path)
        self.img_dir   = img_dir
        self.transform = transform or TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row   = self.df.iloc[i]
        img   = Image.open(f"{self.img_dir}/{row['Image Index']}").convert('RGB')
        label = _encode_label(row['Finding Labels'])
        return self.transform(img), label


def _encode_label(label_str):
    """'Pneumonia|Effusion' → tensor([0,0,1,...,1,...])"""
    vec = torch.zeros(len(DISEASES))
    for d in label_str.split('|'):
        if d in DISEASES:
            vec[DISEASES.index(d)] = 1.0
    return vec


# ── model: DenseNet121 with custom classifier head ────────────────
def build_model():
    """
    DenseNet121 pretrained on ImageNet.
    We replace the final FC layer to output 14 probabilities.
    This is the same architecture used in the original NIH paper.
    """
    model = models.densenet121(weights='IMAGENET1K_V1')
    # Original head: Linear(1024 → 1000). Replace with our 14-class head.
    model.classifier = nn.Linear(1024, len(DISEASES))
    return model


# ── training ──────────────────────────────────────────────────────
def train(model, train_loader, val_loader=None, epochs=5, lr=1e-4):
    """
    Multi-label training: BCEWithLogitsLoss because each image
    can have multiple diseases (not mutually exclusive).
    """
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = model.to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()   # sigmoid + BCE in one stable op

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)                   # raw logits, shape (B, 14)
            loss  = loss_fn(preds, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        if val_loader:
            auc = evaluate(model, val_loader, device)
            print(f"  Val AUC-ROC: {auc:.4f}")

    return model


# ── evaluation ────────────────────────────────────────────────────
def evaluate(model, loader, device=None):
    """Returns mean AUC-ROC across all 14 diseases."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = model(imgs).sigmoid().cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # AUC per disease, then average (skip diseases with no positive samples)
    aucs = []
    for i in range(len(DISEASES)):
        if all_labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
    return float(np.mean(aucs))


# ── inference ─────────────────────────────────────────────────────
def predict(model, img_tensor):
    """
    img_tensor : transformed image tensor, shape (C, H, W)
    Returns    : dict {disease_name: probability}
    """
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))   # add batch dim
        probs  = logits.sigmoid().squeeze()
    return {d: round(p.item(), 3) for d, p in zip(DISEASES, probs)}


# ── quick start ───────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      default='data/Data_Entry_2017.csv')
    parser.add_argument('--img_dir',  default='data/images')
    parser.add_argument('--epochs',   type=int, default=5)
    parser.add_argument('--batch',    type=int, default=32)
    parser.add_argument('--save',     default='cnn_weights.pt')
    args = parser.parse_args()

    dataset    = ChestXrayDataset(args.csv, args.img_dir)
    val_size   = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4)

    model = build_model()
    model = train(model, train_loader, val_loader, epochs=args.epochs)
    torch.save(model.state_dict(), args.save)
    print(f"Saved weights → {args.save}")
