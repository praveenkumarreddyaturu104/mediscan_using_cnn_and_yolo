"""
cnn_classifier.py
-----------------
Classifies chest X-rays as NORMAL or PNEUMONIA using DenseNet121.

Dataset : Chest X-Ray Images (Pneumonia) — Mendeley / Kaggle
          data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/
Download: python download_data.py
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np


# ── transforms ────────────────────────────────────────────────────
# Training: random flips + slight rotation for augmentation
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],    # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

# Val/test: no augmentation, just resize + normalise
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── dataset loader ────────────────────────────────────────────────
def get_loaders(data_dir='data/raw/chest_xray', batch_size=32):
    """
    ImageFolder reads subfolders as class names automatically.
    Folder structure expected:
        data/raw/chest_xray/
            train/NORMAL/       train/PNEUMONIA/
            val/NORMAL/         val/PNEUMONIA/
            test/NORMAL/        test/PNEUMONIA/
    """
    train_ds = ImageFolder(f'{data_dir}/train', transform=TRAIN_TRANSFORM)
    val_ds   = ImageFolder(f'{data_dir}/val',   transform=EVAL_TRANSFORM)
    test_ds  = ImageFolder(f'{data_dir}/test',  transform=EVAL_TRANSFORM)

    print(f"Classes: {train_ds.classes}")          # ['NORMAL', 'PNEUMONIA']
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_ds.classes


# ── model ─────────────────────────────────────────────────────────
def build_model(num_classes=2):
    """
    DenseNet121 pretrained on ImageNet.
    We freeze early layers and only train the classifier head + last dense block.
    This is called fine-tuning — much faster than training from scratch.
    """
    model = models.densenet121(weights='IMAGENET1K_V1')

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last dense block + classifier (the part we retrain)
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True

    # Replace final FC: 1024 features → num_classes outputs
    model.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.3),            # dropout reduces overfitting
        nn.Linear(256, num_classes)
    )
    return model


# ── training ──────────────────────────────────────────────────────
def train(model, train_loader, val_loader, epochs=10, lr=1e-4):
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    model   = model.to(device)

    # Only optimise parameters that require gradients (unfrozen layers)
    opt     = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Learning rate drops by 0.5 every 3 epochs if val loss doesn't improve
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_auc = 0.0
    for epoch in range(epochs):
        # ── train phase ──
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss  = loss_fn(preds, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()

        # ── val phase ──
        auc, acc = evaluate(model, val_loader, device)
        scheduler.step(1 - auc)    # minimise (1 - AUC)

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"loss={running_loss/len(train_loader):.4f} | "
              f"val_AUC={auc:.4f} | val_acc={acc:.4f}")

        # save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'models/cnn_best.pt')
            print(f"  → Saved best model (AUC {auc:.4f})")

    print(f"\nBest val AUC: {best_auc:.4f}")
    return model


# ── evaluation ────────────────────────────────────────────────────
def evaluate(model, loader, device=None):
    """Returns (AUC-ROC, accuracy) on the given loader."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = model(imgs).softmax(dim=1)[:, 1].cpu().numpy()  # prob of PNEUMONIA
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    acc = ((all_probs > 0.5).astype(int) == all_labels).mean()
    return auc, acc


# ── inference on a single image ───────────────────────────────────
def predict_image(model, img_path, classes=['NORMAL', 'PNEUMONIA']):
    """
    Returns dict: {'NORMAL': 0.12, 'PNEUMONIA': 0.88, 'prediction': 'PNEUMONIA'}
    """
    from PIL import Image
    img    = Image.open(img_path).convert('RGB')
    tensor = EVAL_TRANSFORM(img).unsqueeze(0)    # add batch dim
    model.eval()
    with torch.no_grad():
        probs = model(tensor).softmax(dim=1).squeeze()
    result = {cls: round(p.item(), 3) for cls, p in zip(classes, probs)}
    result['prediction'] = classes[probs.argmax().item()]
    return result


# ── entry point ───────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='data/raw/chest_xray')
    parser.add_argument('--epochs',  type=int, default=10)
    parser.add_argument('--batch',   type=int, default=32)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, classes = get_loaders(args.data, args.batch)
    model = build_model(num_classes=len(classes))
    model = train(model, train_loader, val_loader, epochs=args.epochs)

    # final test evaluation
    model.load_state_dict(torch.load('models/cnn_best.pt'))
    auc, acc = evaluate(model, test_loader)
    print(f"\nTest AUC-ROC : {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
