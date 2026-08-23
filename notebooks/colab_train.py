"""
Colab GPU Training Script for TrustworthyMed
Usage in Google Colab:
  1. Upload this file to Colab (left sidebar → Files → Upload)
  2. Run: !python colab_train.py
"""

import os
import sys

# ========== CELL 1: Mount Google Drive ==========
print("=" * 60)
print("STEP 1: Mounting Google Drive...")
print("=" * 60)

from google.colab import drive
drive.mount('/content/drive')

# ========== CELL 2: Clone latest code from GitHub ==========
print("\n" + "=" * 60)
print("STEP 2: Cloning your repo...")
print("=" * 60)

!rm -rf TrustworthyMed
!git clone https://github.com/iqrasafdarr/TrustworthyMed.git
%cd TrustworthyMed

# Switch to your experiments branch (create if not exists)
!git checkout user-experiments 2>/dev/null || git checkout -b user-experiments origin/user-experiments

print("Latest code pulled!")

# ========== CELL 3: Install dependencies ==========
print("\n" + "=" * 60)
print("STEP 3: Installing packages...")
print("=" * 60)

!pip install -q torch torchvision scikit-learn matplotlib seaborn pandas numpy pillow tqdm pyyaml opencv-python

# ========== CELL 4: Link data from Google Drive ==========
print("\n" + "=" * 60)
print("STEP 4: Linking datasets from Drive...")
print("=" * 60)

# Create folders
!mkdir -p data/raw/ham10000/images
!mkdir -p data/raw/bcn20000/images
!mkdir -p data/splits

# --- METHOD 1: Try ZIP file first (fastest) ---
ZIP_PATH = "/content/drive/MyDrive/TrustworthyMed_data.zip"
if os.path.exists(ZIP_PATH):
    print("Found ZIP file! Extracting...")
    !unzip -q {ZIP_PATH} -d /content/extracted/
    !cp -r /content/extracted/data/* data/
    print("ZIP extracted!")
else:
    print("No ZIP found. Trying individual files...")
    # --- METHOD 2: Individual files ---
    !cp /content/drive/MyDrive/TrustworthyMed/data/ham10000/HAM10000_metadata.csv data/raw/ham10000/ 2>/dev/null || echo "HAM metadata missing"
    !cp -r /content/drive/MyDrive/TrustworthyMed/data/ham10000/images/* data/raw/ham10000/images/ 2>/dev/null || echo "HAM images missing"
    !cp /content/drive/MyDrive/TrustworthyMed/data/bcn20000/bcn20000_metadata.csv data/raw/bcn20000/ 2>/dev/null || echo "BCN metadata missing"
    !cp -r /content/drive/MyDrive/TrustworthyMed/data/bcn20000/images/* data/raw/bcn20000/images/ 2>/dev/null || echo "BCN images missing"

# Check what we got
ham_imgs = len(os.listdir('data/raw/ham10000/images')) if os.path.exists('data/raw/ham10000/images') else 0
bcn_imgs = len(os.listdir('data/raw/bcn20000/images')) if os.path.exists('data/raw/bcn20000/images') else 0
print(f"\nHAM10000 images: {ham_imgs}")
print(f"BCN20000 images: {bcn_imgs}")

if ham_imgs < 1000:
    print("ERROR: HAM10000 images not found! Check your Drive path.")
    sys.exit(1)

# ========== CELL 5: Setup splits ==========
print("\n" + "=" * 60)
print("STEP 5: Creating data splits...")
print("=" * 60)

!python scripts/setup_real_data.py

# ========== CELL 6: GPU Check ==========
print("\n" + "=" * 60)
print("STEP 6: GPU Check...")
print("=" * 60)

!nvidia-smi

# ========== CELL 7: Update config for GPU ==========
print("\n" + "=" * 60)
print("STEP 7: Updating config for GPU...")
print("=" * 60)

import yaml
with open('configs/experiment_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['training']['device'] = 'cuda'
config['training']['num_epochs'] = 20
config['training']['batch_size'] = 32
config['training']['num_workers'] = 2
config['logging']['use_wandb'] = False

with open('configs/experiment_config.yaml', 'w') as f:
    yaml.dump(config, f)

print("Config updated!")

# ========== CELL 8: TRAIN (GPU!) ==========
print("\n" + "=" * 60)
print("STEP 8: Starting training on GPU...")
print("=" * 60)

# Create Drive results folder for auto-save
!mkdir -p /content/drive/MyDrive/TrustworthyMed/results

# Run training
!python scripts/train_baseline.py --config configs/experiment_config.yaml

# ========== CELL 9: Save results back to Drive ==========
print("\n" + "=" * 60)
print("STEP 9: Saving results to Google Drive...")
print("=" * 60)

# Copy all results
!cp -r results/* /content/drive/MyDrive/TrustworthyMed/results/ 2>/dev/null || echo "No new results"

# Zip results for easy download
!zip -r /content/drive/MyDrive/TrustworthyMed/results_backup.zip results/ 2>/dev/null || echo "Could not zip results"

print("\n" + "=" * 60)
print("DONE! Results saved to:")
print("  - Google Drive: TrustworthyMed/results/")
print("  - Google Drive: TrustworthyMed/results_backup.zip")
print("=" * 60)