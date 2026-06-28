"""
utils/helpers.py
----------------
Shared utility functions used by cnn_classifier, yolo_detector, pipeline.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ── device helper ─────────────────────────────────────────────────
def get_device():
    """Returns GPU if available, else CPU. Prints what it picked."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    return device


# ── visualise predictions on a grid of images ─────────────────────
def show_predictions(model, loader, classes, n=16, save_path=None):
    """
    Shows n sample images with predicted vs true labels.
    Green title = correct, Red = wrong.
    """
    device = get_device()
    model.eval()
    images, labels, preds = [], [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out  = model(imgs).softmax(dim=1).argmax(dim=1).cpu()
            images.extend(imgs.cpu())
            labels.extend(lbls)
            preds.extend(out)
            if len(images) >= n:
                break

    # denormalise for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    cols = 4
    rows = (min(n, len(images)) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()

    for i in range(min(n, len(images))):
        img = (images[i] * std + mean).clamp(0,1).permute(1,2,0).numpy()
        axes[i].imshow(img, cmap='gray' if img.mean() < 0.5 else None)
        correct  = labels[i] == preds[i]
        color    = 'green' if correct else 'red'
        axes[i].set_title(
            f"True: {classes[labels[i]]}\nPred: {classes[preds[i]]}",
            color=color, fontsize=8
        )
        axes[i].axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
        print(f"Saved prediction grid → {save_path}")
    else:
        plt.show()
    plt.close()


# ── plot training curves ───────────────────────────────────────────
def plot_history(losses, aucs, save_path='outputs/training_curves.png'):
    """Plots loss and AUC curves side by side after training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(losses, color='#E24B4A', linewidth=2)
    ax1.set_title('Training loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.grid(alpha=0.3)

    ax2.plot(aucs, color='#1D9E75', linewidth=2)
    ax2.set_title('Validation AUC-ROC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_ylim(0.5, 1.0)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"Training curves → {save_path}")
    plt.close()


# ── count dataset class balance ───────────────────────────────────
def class_counts(data_dir):
    """Prints how many images per class (helps spot imbalance)."""
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        print(f"\n{split}/")
        for cls in sorted(os.listdir(split_dir)):
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path):
                count = len(os.listdir(cls_path))
                print(f"  {cls:<15} {count:>5} images")
