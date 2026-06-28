"""
yolo_detector.py
----------------
Fine-tunes YOLOv8 on NIH bounding box annotations to locate lesions.
Dataset file needed: BBox_List_2017.csv  (comes with NIH download)

Columns: Image Index, Finding Label, Bbox [x, y, w, h]  (pixel coords, 1024×1024)
"""

import os
import yaml
import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# ── step 1: convert NIH pixel coords → YOLO normalised format ─────
def convert_nih_to_yolo(
    bbox_csv   : str = 'data/BBox_List_2017.csv',
    img_dir    : str = 'data/images',
    out_dir    : str = 'data/yolo',
    img_size   : int = 1024,
    val_split  : float = 0.15,
):
    """
    NIH format  : x_top_left, y_top_left, width, height  (pixels)
    YOLO format : class  cx  cy  w  h                    (0-1 normalised)

    Creates:
        data/yolo/images/train/  data/yolo/images/val/
        data/yolo/labels/train/  data/yolo/labels/val/
    Returns list of class names (needed for dataset.yaml).
    """
    df      = pd.read_csv(bbox_csv)
    classes = sorted(df['Finding Label'].unique().tolist())

    for split in ['train', 'val']:
        os.makedirs(f'{out_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{out_dir}/labels/{split}',  exist_ok=True)

    # shuffle and split by image (not by row)
    images  = df['Image Index'].unique()
    n_val   = int(len(images) * val_split)
    val_set = set(images[:n_val])

    for _, row in df.iterrows():
        img_name = row['Image Index']
        split    = 'val' if img_name in val_set else 'train'

        # normalise box coords to [0, 1]
        x_tl = row['Bbox [x']        # top-left x
        y_tl = row['y]']             # top-left y  (column name has typo in NIH csv)
        w    = row['w']
        h    = row['h']
        cx   = (x_tl + w / 2) / img_size
        cy   = (y_tl + h / 2) / img_size
        wn   = w / img_size
        hn   = h / img_size
        cls  = classes.index(row['Finding Label'])

        # append label line  (one line per box per image)
        label_file = f"{out_dir}/labels/{split}/{img_name.replace('.png', '.txt')}"
        with open(label_file, 'a') as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")

        # symlink or copy image so YOLO can find it
        src = f"{img_dir}/{img_name}"
        dst = f"{out_dir}/images/{split}/{img_name}"
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    print(f"Converted {len(df)} boxes across {len(images)} images.")
    print(f"Classes ({len(classes)}): {classes}")
    return classes


# ── step 2: write dataset.yaml that YOLOv8 reads ──────────────────
def make_dataset_yaml(
    classes   : list,
    yolo_dir  : str = 'data/yolo',
    yaml_path : str = 'chest.yaml',
):
    """
    YOLOv8 needs a yaml file that tells it where images/labels are
    and what the class names are.
    """
    cfg = {
        'path'  : os.path.abspath(yolo_dir),
        'train' : 'images/train',
        'val'   : 'images/val',
        'nc'    : len(classes),
        'names' : classes,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Dataset yaml saved → {yaml_path}")
    return yaml_path


# ── step 3: fine-tune YOLOv8 ──────────────────────────────────────
def train_yolo(
    yaml_path : str = 'chest.yaml',
    epochs    : int = 50,
    imgsz     : int = 640,
    batch     : int = 16,
):
    """
    Starts from YOLOv8s pretrained weights (auto-downloaded ~22MB).
    'patience=10' stops early if val mAP doesn't improve for 10 epochs.
    Best weights saved to: runs/detect/train/weights/best.pt
    """
    model = YOLO('yolov8s.pt')          # s=small: good speed/accuracy balance
    model.train(
        data      = yaml_path,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        patience  = 10,                 # early stopping
        save      = True,
        project   = 'runs/detect',
        name      = 'mediscan',
    )
    # Print validation mAP after training
    metrics = model.val()
    print(f"mAP@0.5 : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    return model


# ── step 4: detect lesions in a single image ──────────────────────
def detect_lesions(
    img_path    : str,
    model_path  : str = 'runs/detect/mediscan/weights/best.pt',
    conf_thresh : float = 0.4,
):
    """
    Returns a list of dicts:
        [{'class': 'Effusion', 'conf': 0.73, 'box': [x1,y1,x2,y2]}, ...]
    box coords are in pixels relative to the original image size.
    """
    model   = YOLO(model_path)
    results = model.predict(img_path, conf=conf_thresh, verbose=False)

    detections = []
    for box in results[0].boxes:
        detections.append({
            'class' : results[0].names[int(box.cls)],
            'conf'  : round(float(box.conf), 3),
            'box'   : [round(v, 1) for v in box.xyxy[0].tolist()],  # [x1,y1,x2,y2]
        })
    return detections


# ── quick start ───────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_csv', default='data/BBox_List_2017.csv')
    parser.add_argument('--img_dir',  default='data/images')
    parser.add_argument('--epochs',   type=int, default=50)
    args = parser.parse_args()

    # 1. Convert labels
    classes = convert_nih_to_yolo(args.bbox_csv, args.img_dir)

    # 2. Write yaml
    yaml_path = make_dataset_yaml(classes)

    # 3. Train
    train_yolo(yaml_path, epochs=args.epochs)
