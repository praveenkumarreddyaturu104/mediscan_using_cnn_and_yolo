"""
yolo_detector.py
----------------
Fine-tunes YOLOv8 on VinBigData chest X-ray dataset.
14 abnormality classes, bounding boxes included, images resized to 512×512.

Dataset: vinbigdata-chest-xray-resized-png-512x512 (Kaggle, free)
Download: python download_data.py --kaggle
"""

import os
import yaml
import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# 14 abnormality classes in VinBigData
CLASSES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification',
    'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration',
    'Lung Opacity', 'Nodule/Mass', 'Other lesion',
    'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
    'Pulmonary fibrosis'
]

# class_id 14 = "No finding" in VinBigData — we skip it (no box to draw)
NO_FINDING_ID = 14
IMG_SIZE = 512   # VinBigData resized version is 512×512


# ── step 1: convert VinBigData CSV → YOLO label files ─────────────
def convert_vinbig_to_yolo(
    train_csv : str = 'data/raw/vinbigdata/train.csv',
    img_dir   : str = 'data/raw/vinbigdata/train',
    out_dir   : str = 'data/yolo',
    val_split : float = 0.15,
):
    """
    VinBigData CSV columns:
        image_id, class_name, class_id, x_min, y_min, x_max, y_max, rad_id

    Converts pixel [x_min,y_min,x_max,y_max] → YOLO [cx,cy,w,h] normalised 0-1.
    Skips class_id == 14 (No finding — no bounding box).
    """
    df = pd.read_csv(train_csv)
    df = df[df['class_id'] != NO_FINDING_ID]   # drop "No finding" rows

    # split by image (not by row)
    image_ids = df['image_id'].unique()
    n_val     = int(len(image_ids) * val_split)
    val_ids   = set(image_ids[:n_val])

    for split in ['train', 'val']:
        os.makedirs(f'{out_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{out_dir}/labels/{split}',  exist_ok=True)

    converted, skipped = 0, 0
    for _, row in df.iterrows():
        img_id = row['image_id']
        split  = 'val' if img_id in val_ids else 'train'

        # convert to YOLO format (normalise by image size)
        x_min, y_min = row['x_min'], row['y_min']
        x_max, y_max = row['x_max'], row['y_max']

        # guard: skip degenerate boxes
        if x_max <= x_min or y_max <= y_min:
            skipped += 1
            continue

        cx = ((x_min + x_max) / 2) / IMG_SIZE
        cy = ((y_min + y_max) / 2) / IMG_SIZE
        w  = (x_max - x_min) / IMG_SIZE
        h  = (y_max - y_min) / IMG_SIZE
        cls = int(row['class_id'])

        # write label line (appends — one image can have multiple boxes)
        label_path = f"{out_dir}/labels/{split}/{img_id}.txt"
        with open(label_path, 'a') as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # copy image to yolo folder (only once per image)
        src = f"{img_dir}/{img_id}.png"
        dst = f"{out_dir}/images/{split}/{img_id}.png"
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy(src, dst)

        converted += 1

    print(f"Converted {converted} boxes  |  Skipped {skipped} degenerate boxes")
    return CLASSES


# ── step 2: write dataset.yaml ────────────────────────────────────
def make_dataset_yaml(
    classes   : list = CLASSES,
    yolo_dir  : str  = 'data/yolo',
    yaml_path : str  = 'data/vinbig.yaml',
):
    cfg = {
        'path'  : os.path.abspath(yolo_dir),
        'train' : 'images/train',
        'val'   : 'images/val',
        'nc'    : len(classes),
        'names' : classes,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Dataset yaml → {yaml_path}")
    return yaml_path


# ── step 3: fine-tune YOLOv8 ──────────────────────────────────────
def train_yolo(
    yaml_path : str = 'data/vinbig.yaml',
    epochs    : int = 50,
    imgsz     : int = 512,
    batch     : int = 16,
):
    """
    Starts from YOLOv8s pretrained weights (auto-downloaded ~22 MB).
    Results saved to: models/yolo_best.pt  (we copy after training)
    """
    model = YOLO('yolov8s.pt')
    model.train(
        data     = yaml_path,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        patience = 10,           # early stop if mAP stalls
        project  = 'runs',
        name     = 'mediscan_yolo',
        save     = True,
    )

    # copy best weights to models/
    best_src = 'runs/mediscan_yolo/weights/best.pt'
    os.makedirs('models', exist_ok=True)
    shutil.copy(best_src, 'models/yolo_best.pt')

    metrics = model.val()
    print(f"\nmAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    return model


# ── step 4: detect lesions in one image ───────────────────────────
def detect_lesions(
    img_path    : str,
    model_path  : str   = 'models/yolo_best.pt',
    conf_thresh : float = 0.4,
):
    """
    Returns list of dicts:
        [{'class': 'Cardiomegaly', 'conf': 0.81, 'box': [x1,y1,x2,y2]}, ...]
    """
    model   = YOLO(model_path)
    results = model.predict(img_path, conf=conf_thresh, verbose=False)

    return [{
        'class' : results[0].names[int(b.cls)],
        'conf'  : round(float(b.conf), 3),
        'box'   : [round(v, 1) for v in b.xyxy[0].tolist()],
    } for b in results[0].boxes]


# ── entry point ───────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',      default='data/raw/vinbigdata/train.csv')
    parser.add_argument('--img_dir',  default='data/raw/vinbigdata/train')
    parser.add_argument('--epochs',   type=int, default=50)
    args = parser.parse_args()

    classes   = convert_vinbig_to_yolo(args.csv, args.img_dir)
    yaml_path = make_dataset_yaml(classes)
    train_yolo(yaml_path, epochs=args.epochs)
