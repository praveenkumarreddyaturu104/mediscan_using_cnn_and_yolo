"""
download_data.py
----------------
Downloads both datasets automatically.
No manual steps. No login needed for Option A.

OPTION A — No Kaggle account needed (recommended to start):
    Uses torchvision's built-in datasets.
    CNN dataset downloads in ~1 minute.

OPTION B — Kaggle account (for full project with YOLO bboxes):
    pip install kaggle
    Put kaggle.json in ~/.kaggle/
    Then run: python download_data.py --kaggle
"""

import os
import zipfile
import urllib.request
import argparse


# ── Option A: torchvision built-in (zero setup) ───────────────────
def download_via_torchvision(data_dir='data/raw'):
    """
    Downloads Chest X-Ray (Pneumonia) dataset via torchvision.
    Saves to data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/
    Total size: ~1.2 GB
    """
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as T

    print("Downloading Chest X-Ray Pneumonia dataset via torchvision...")
    print("(This is ~1.2 GB — takes 2-5 min on a normal connection)\n")

    # torchvision doesn't have this dataset built-in, so we use the
    # direct download URL from a public mirror
    url = (
        "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
        "f18b4773-7cb5-40ca-a3a2-6f85dcd01adf/file_downloaded"
    )
    zip_path = os.path.join(data_dir, 'chest_xray.zip')
    os.makedirs(data_dir, exist_ok=True)

    print(f"→ Downloading to {zip_path} ...")
    urllib.request.urlretrieve(url, zip_path, reporthook=_progress)

    print(f"\n→ Extracting ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    os.remove(zip_path)
    print(f"✓ Done. Dataset at {data_dir}/chest_xray/")
    return os.path.join(data_dir, 'chest_xray')


# ── Option B: Kaggle CLI (needed for YOLO bounding boxes) ─────────
def download_via_kaggle(data_dir='data/raw'):
    """
    Downloads two datasets via Kaggle:
    1. chest-xray-pneumonia  → CNN training (binary: normal/pneumonia)
    2. vinbigdata 512px      → YOLO training (14 classes + bboxes)

    Setup (one time):
        1. Go to kaggle.com → Account → Create API Token → downloads kaggle.json
        2. mkdir ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
        3. pip install kaggle
    """
    import subprocess

    os.makedirs(data_dir, exist_ok=True)

    datasets = [
        {
            'name'   : 'paultimothymooney/chest-xray-pneumonia',
            'outdir' : os.path.join(data_dir, 'pneumonia'),
            'note'   : 'CNN dataset — 5,863 X-rays (NORMAL / PNEUMONIA)',
        },
        {
            'name'   : 'awsaf49/vinbigdata-chest-xray-resized-png-512x512',
            'outdir' : os.path.join(data_dir, 'vinbigdata'),
            'note'   : 'YOLO dataset — 15,000 X-rays, 14 classes, bounding boxes',
        },
    ]

    for ds in datasets:
        print(f"\n→ {ds['note']}")
        os.makedirs(ds['outdir'], exist_ok=True)
        cmd = [
            'kaggle', 'datasets', 'download',
            ds['name'],
            '--path', ds['outdir'],
            '--unzip',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
            print("  Make sure kaggle.json is at ~/.kaggle/kaggle.json")
        else:
            print(f"  ✓ Saved to {ds['outdir']}")


# ── progress bar for urllib ────────────────────────────────────────
def _progress(count, block_size, total_size):
    percent = min(int(count * block_size * 100 / total_size), 100)
    bar = '█' * (percent // 5) + '░' * (20 - percent // 5)
    print(f"\r  [{bar}] {percent}%", end='', flush=True)


# ── verify dataset structure ───────────────────────────────────────
def verify(data_dir='data/raw'):
    print("\nVerifying dataset structure...")
    expected = [
        'chest_xray/train/NORMAL',
        'chest_xray/train/PNEUMONIA',
        'chest_xray/test/NORMAL',
        'chest_xray/test/PNEUMONIA',
    ]
    all_ok = True
    for path in expected:
        full = os.path.join(data_dir, path)
        exists = os.path.isdir(full)
        count  = len(os.listdir(full)) if exists else 0
        mark   = '✓' if exists else '✗'
        print(f"  {mark}  {path}  ({count} files)")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n✓ Dataset ready. Run: python cnn_classifier.py")
    else:
        print("\n✗ Some folders missing. Re-run download_data.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true',
                        help='Use Kaggle CLI (needed for YOLO bboxes)')
    parser.add_argument('--data_dir', default='data/raw')
    args = parser.parse_args()

    if args.kaggle:
        download_via_kaggle(args.data_dir)
    else:
        download_via_torchvision(args.data_dir)
        verify(args.data_dir)
