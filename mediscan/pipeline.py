"""
pipeline.py
-----------
Runs CNN → YOLO in sequence on one chest X-ray.
CNN: is this NORMAL or PNEUMONIA?
YOLO: draw a box around the abnormal region.

Usage:
    python pipeline.py --img data/raw/chest_xray/test/PNEUMONIA/person1_virus_006.jpeg
"""

import argparse
import torch
from PIL import Image, ImageDraw
from cnn_classifier import build_model, predict_image, EVAL_TRANSFORM
from yolo_detector  import detect_lesions

CNN_WEIGHTS  = 'models/cnn_best.pt'
YOLO_WEIGHTS = 'models/yolo_best.pt'
CNN_THRESHOLD = 0.6   # only run YOLO if CNN predicts PNEUMONIA ≥ 60%


# ── load CNN once ─────────────────────────────────────────────────
def load_cnn(weights=CNN_WEIGHTS):
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(weights, map_location='cpu'))
    model.eval()
    return model


# ── main analysis ─────────────────────────────────────────────────
def analyse(img_path, cnn_model, yolo_weights=YOLO_WEIGHTS):
    """
    Returns:
        {
          'cnn'       : {'NORMAL': 0.08, 'PNEUMONIA': 0.92, 'prediction': 'PNEUMONIA'},
          'run_yolo'  : True,
          'detections': [{'class': ..., 'conf': ..., 'box': [...]}]
        }
    """
    # step 1 — CNN classification
    cnn_result = predict_image(cnn_model, img_path)

    # step 2 — gate: only run expensive YOLO if CNN is confident
    pneumonia_prob = cnn_result.get('PNEUMONIA', 0)
    run_yolo       = pneumonia_prob >= CNN_THRESHOLD

    detections = []
    if run_yolo:
        try:
            detections = detect_lesions(img_path, yolo_weights)
        except Exception as e:
            print(f"  YOLO skipped (weights not found or error): {e}")

    return {
        'image'      : img_path,
        'cnn'        : cnn_result,
        'run_yolo'   : run_yolo,
        'detections' : detections,
    }


# ── annotate image with boxes ─────────────────────────────────────
def draw_report(result, out_path='outputs/report.png'):
    """Draws CNN result + YOLO boxes on the original image."""
    os.makedirs('outputs', exist_ok=True)
    img  = Image.open(result['image']).convert('RGB').resize((512, 512))
    draw = ImageDraw.Draw(img)

    # CNN label at top
    cnn_text = (f"CNN: {result['cnn']['prediction']}  "
                f"({result['cnn'].get('PNEUMONIA', 0):.0%} pneumonia)")
    draw.rectangle([0, 0, img.width, 22], fill='navy')
    draw.text((6, 4), cnn_text, fill='white')

    # YOLO boxes
    for det in result['detections']:
        x1, y1, x2, y2 = det['box']
        # scale to 512 if needed (YOLO was trained on 512)
        label = f"{det['class']}  {det['conf']:.0%}"
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        tw, th = draw.textbbox((0, 0), label)[2:4]
        draw.rectangle([x1, max(0, y1-th-4), x1+tw+6, y1], fill='red')
        draw.text((x1+3, max(0, y1-th-2)), label, fill='white')

    img.save(out_path)
    return img


# ── pretty print ──────────────────────────────────────────────────
def print_summary(result):
    print('\n' + '─'*52)
    print(f"Image : {result['image']}")
    print('─'*52)
    cnn = result['cnn']
    print(f"\nCNN prediction : {cnn['prediction']}")
    for cls, prob in cnn.items():
        if cls == 'prediction': continue
        bar = '█' * int(prob * 25)
        print(f"  {cls:<12} {prob:.3f}  {bar}")

    if result['run_yolo']:
        print(f"\nYOLO ran  ({len(result['detections'])} detection(s)):")
        for d in result['detections']:
            x1,y1,x2,y2 = d['box']
            print(f"  {d['class']:<25} conf={d['conf']:.3f}  "
                  f"box=[{x1:.0f},{y1:.0f} → {x2:.0f},{y2:.0f}]")
        if not result['detections']:
            print("  No boxes above confidence threshold.")
    else:
        print(f"\nYOLO skipped "
              f"(CNN pneumonia score {result['cnn'].get('PNEUMONIA',0):.2f} "
              f"< threshold {CNN_THRESHOLD})")
    print('─'*52 + '\n')


# ── entry point ───────────────────────────────────────────────────
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',  required=True)
    parser.add_argument('--cnn',  default=CNN_WEIGHTS)
    parser.add_argument('--yolo', default=YOLO_WEIGHTS)
    parser.add_argument('--out',  default='outputs/report.png')
    args = parser.parse_args()

    print("Loading CNN model...")
    cnn = load_cnn(args.cnn)

    print("Running pipeline...")
    result = analyse(args.img, cnn, args.yolo)

    print_summary(result)
    draw_report(result, args.out)
    print(f"Saved annotated report → {args.out}")
