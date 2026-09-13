import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path

DATASET_SLUG = "thoughtvector/customer-support-on-twitter"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def download_via_kaggle_cli():
    print(f"Attempting download via Kaggle CLI: {DATASET_SLUG}...")
    import subprocess
    cmd = ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(RAW_DIR), "--unzip"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("Successfully downloaded and extracted via Kaggle CLI.")
        return True
    else:
        print("Kaggle CLI failed or not configured:", res.stderr.strip())
        return False

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = RAW_DIR / "twcs.csv"
    if target_csv.exists():
        size_mb = target_csv.stat().st_size / (1024 * 1024)
        print(f"Dataset already exists at {target_csv} ({size_mb:.2f} MB).")
        return

    print("Checking for Kaggle API credentials...")
    kaggle_success = download_via_kaggle_cli()
    if not kaggle_success:
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("The full dataset 'Customer Support on Twitter' is hosted on Kaggle:")
        print("https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        print("\nLicense: CC BY-NC-SA 4.0")
        print("To download automatically, place your kaggle.json in ~/.kaggle/ and rerun this script.")
        print(f"Alternatively, download twcs.csv and place it at: {target_csv.resolve()}")
        print("For reproducible demonstration & testing without downloading 700MB+, run:")
        print("  python -m pipeline.prepare_demo_data")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()
