"""
pipeline.py
-----------
Runs CNN classifier → YOLO detector in sequence on a chest X-ray.
CNN answers "what disease?" — YOLO answers "where is it?"

Usage:
    python pipeline.py --img data/images/00000013_005.png
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from cnn_classifier import build_model, predict, DISEASES
from yolo_detector  import detect_lesions

# ── config ────────────────────────────────────────────────────────
CNN_WEIGHTS  = 'cnn_weights.pt'
YOLO_WEIGHTS = 'runs/detect/mediscan/weights/best.pt'
CNN_THRESHOLD = 0.5   # run YOLO only if CNN scores above this

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ── load CNN once at startup ───────────────────────────────────────
def load_cnn(weights_path=CNN_WEIGHTS):
    model = build_model()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


# ── main pipeline function ─────────────────────────────────────────
def analyse_xray(img_path, cnn_model, yolo_weights=YOLO_WEIGHTS):
    """
    Step 1: CNN scores all 14 diseases from the full image.
    Step 2: If any score > CNN_THRESHOLD, run YOLO for localisation.
    Step 3: Return structured result dict.
    """
    # step 1 — classify
    img    = Image.open(img_path).convert('RGB')
    tensor = TRANSFORM(img)
    probs  = predict(cnn_model, tensor)           # {disease: 0.0-1.0}

    # step 2 — filter to confident findings
    findings = {d: p for d, p in probs.items() if p > CNN_THRESHOLD}

    result = {
        'image'        : img_path,
        'all_probs'    : probs,          # full score table
        'cnn_findings' : findings,       # diseases above threshold
        'detections'   : [],             # YOLO boxes (filled next)
    }

    # step 3 — localise only if CNN found something (saves compute)
    if findings:
        result['detections'] = detect_lesions(img_path, yolo_weights)

    return result


# ── draw annotated report image ────────────────────────────────────
def draw_report(result, out_path='report.png'):
    """
    Draws red bounding boxes + labels on the original X-ray.
    Saves to out_path and returns the PIL Image.
    """
    img  = Image.open(result['image']).convert('RGB')
    draw = ImageDraw.Draw(img)

    for det in result['detections']:
        x1, y1, x2, y2 = det['box']
        label = f"{det['class']}  {det['conf']:.0%}"

        # box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # label background + text
        tw, th = draw.textbbox((0, 0), label)[2:4]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill='red')
        draw.text((x1 + 3, y1 - th - 2), label, fill='white')

    img.save(out_path)
    print(f"Annotated report saved → {out_path}")
    return img


# ── pretty print summary ───────────────────────────────────────────
def print_report(result):
    print("\n" + "─" * 50)
    print(f"Image : {result['image']}")
    print("─" * 50)

    print("\nCNN findings (above threshold):")
    if result['cnn_findings']:
        for disease, prob in sorted(result['cnn_findings'].items(),
                                    key=lambda x: -x[1]):
            bar = '█' * int(prob * 20)
            print(f"  {disease:<25}  {prob:.3f}  {bar}")
    else:
        print("  No findings above threshold — likely normal.")

    print("\nYOLO detections (bounding boxes):")
    if result['detections']:
        for d in result['detections']:
            x1,y1,x2,y2 = d['box']
            print(f"  {d['class']:<20}  conf={d['conf']:.3f}  "
                  f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    else:
        print("  No boxes drawn (CNN below threshold or no YOLO detections).")
    print("─" * 50 + "\n")


# ── entry point ───────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MediScan X-ray analyser')
    parser.add_argument('--img',   required=True,    help='Path to chest X-ray PNG')
    parser.add_argument('--cnn',   default=CNN_WEIGHTS,  help='CNN weights file')
    parser.add_argument('--yolo',  default=YOLO_WEIGHTS, help='YOLO weights file')
    parser.add_argument('--out',   default='report.png', help='Output annotated image')
    args = parser.parse_args()

    print("Loading CNN...")
    cnn = load_cnn(args.cnn)

    print("Running pipeline...")
    result = analyse_xray(args.img, cnn, args.yolo)

    print_report(result)
    draw_report(result, args.out)
