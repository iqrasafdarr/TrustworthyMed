"""
Deep ensemble baseline for TrustworthyMed.

Trains N ResNet-50 classifiers (same architecture, different random seeds),
then uses prediction variance across the ensemble as an uncertainty score.
Computes AUROC of that score for detecting misclassified melanomas, to
compare directly against ECS (0.531) and entropy (0.522).

Reference: Lakshminarayanan, Pritzel, Blundell, "Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles," NeurIPS 2017.

USAGE:
  1. First run with TIMING_TEST_ONLY = True to measure one model's training
     time before committing to the full ensemble.
  2. Once you know your budget, set N_ENSEMBLE (3 recommended given time
     constraints) and TIMING_TEST_ONLY = False, then run for real.

Run from your project root:
    python -m scripts.train_deep_ensemble
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from src.models.baselines import SkinLesionClassifier

# ==================== CONFIG ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SAFETY SWITCH: set True first to time a single model before committing ---
TIMING_TEST_ONLY = True

N_ENSEMBLE = 3  # 3 is a legitimate, citable ensemble size; use 5 only if time allows
MAX_EPOCHS = 30  # reduced from your original 50 — early stopping will likely cut this short anyway
PATIENCE = 5      # slightly tighter than your original 7, given the time budget
BATCH_SIZE = 32
LR = 1e-4

# --- Point this at your real combined splits file ---
SPLITS_CSV = "data/splits/ham10000_hospital_splits.csv"
HAM_IMG_DIR = "data/raw/ham10000/images"
CHECKPOINT_DIR = "results/ensemble_checkpoints"
MELANOMA_RESULTS_CSV = "results/ecs_melanoma_results_with_stats_v2.csv"  # your existing melanoma cohort file
OUTPUT_TXT = "results/ensemble_evaluation.txt"

# HAM10000's 7 diagnosis codes, alphabetically ordered — this MUST match the class index
# order your original baseline model was trained with. If your original training script
# used a different order (check train_baseline.py or your config for the class list),
# update this list to match exactly, or the ensemble's predictions will be scored against
# the wrong classes.
DX_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DX_TO_IDX = {dx: i for i, dx in enumerate(DX_CLASSES)}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class HAMDataset(torch.utils.data.Dataset):
    """
    Loads from the combined splits CSV (lesion_id, image_id, dx, dx_type, age, sex,
    localization, dataset, split), filtering to the requested split value and
    encoding the string 'dx' diagnosis code into a class index via DX_TO_IDX.
    """
    def __init__(self, csv_path, split_value, img_dir, tf):
        full_df = pd.read_csv(csv_path)
        if 'split' not in full_df.columns:
            raise ValueError(f"Expected a 'split' column in {csv_path}. "
                              f"Columns found: {full_df.columns.tolist()}")
        self.df = full_df[full_df['split'] == split_value].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found with split == '{split_value}' in {csv_path}. "
                              f"Unique split values present: {full_df['split'].unique().tolist()}")
        self.img_dir = img_dir
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        label = DX_TO_IDX[row['dx']]
        return img, label


def train_one_model(seed, train_loader, val_loader):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SkinLesionClassifier(model_name='resnet50', num_classes=7, pretrained=True, dropout=0.5)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                n_val += imgs.size(0)
        val_loss /= n_val

        print(f"    [seed {seed}] epoch {epoch+1}/{MAX_EPOCHS}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"    [seed {seed}] early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def get_ensemble_predictions(models, img_path):
    """Returns softmax probs from each ensemble member for one image."""
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    probs_list = []
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy()[0])
    return np.stack(probs_list)  # shape: (n_ensemble, n_classes)


def main():
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    if not os.path.isfile(SPLITS_CSV):
        raise FileNotFoundError(
            f"SPLITS_CSV not found: {SPLITS_CSV}. Update this path at the top of the script "
            f"to point at your real combined splits CSV before running."
        )

    print("Loading datasets...")
    train_ds = HAMDataset(SPLITS_CSV, 'train', HAM_IMG_DIR, train_transform)
    val_ds = HAMDataset(SPLITS_CSV, 'val', HAM_IMG_DIR, transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"  train n={len(train_ds)}, val n={len(val_ds)}")

    n_to_train = 1 if TIMING_TEST_ONLY else N_ENSEMBLE
    if TIMING_TEST_ONLY:
        print("\n*** TIMING TEST MODE: training 1 model to measure real wall-clock time. ***")
        print("*** Once done, decide N_ENSEMBLE and set TIMING_TEST_ONLY = False. ***\n")

    models = []
    timings = []
    for i in range(n_to_train):
        seed = 100 + i
        print(f"\nTraining ensemble member {i+1}/{n_to_train} (seed={seed})...")
        t0 = time.time()
        model = train_one_model(seed, train_loader, val_loader)
        elapsed = time.time() - t0
        timings.append(elapsed)
        print(f"  Member {i+1} done in {elapsed/60:.1f} minutes")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ensemble_seed{seed}.pth")
        torch.save(model.state_dict(), ckpt_path)
        models.append(model)

    if TIMING_TEST_ONLY:
        est_full = timings[0] * N_ENSEMBLE / 60
        print(f"\n{'='*60}")
        print(f"TIMING RESULT: one model took {timings[0]/60:.1f} minutes.")
        print(f"Estimated time for full {N_ENSEMBLE}-model ensemble: ~{est_full:.1f} minutes "
              f"({est_full/60:.1f} hours)")
        print(f"{'='*60}")
        print("\nDecide now: if that estimate fits your remaining time, set "
              "TIMING_TEST_ONLY = False and rerun. Otherwise, use the honest-limitation "
              "fallback sentence instead of attempting the full ensemble.")
        return

    # ==================== EVALUATE ENSEMBLE ON MELANOMA COHORT ====================
    print(f"\n{'='*60}")
    print("EVALUATING ENSEMBLE ON MELANOMA VALIDATION COHORT")
    print(f"{'='*60}")

    if not os.path.isfile(MELANOMA_RESULTS_CSV):
        raise FileNotFoundError(f"MELANOMA_RESULTS_CSV not found: {MELANOMA_RESULTS_CSV}")

    df = pd.read_csv(MELANOMA_RESULTS_CSV)
    if 'image_id' not in df.columns or 'is_correct' not in df.columns:
        raise ValueError(f"Expected 'image_id' and 'is_correct' columns. Found: {df.columns.tolist()}")

    ensemble_uncertainty = []
    missing = []
    for i, row in df.iterrows():
        img_path = os.path.join(HAM_IMG_DIR, f"{row['image_id']}.jpg")
        if not os.path.exists(img_path):
            missing.append(row['image_id'])
            ensemble_uncertainty.append(None)
            continue
        probs = get_ensemble_predictions(models, img_path)  # (n_ensemble, n_classes)
        # Predictive variance: mean of per-class variance across ensemble members
        variance = probs.var(axis=0).mean()
        ensemble_uncertainty.append(variance)

    if missing:
        print(f"\nWARNING: {len(missing)} images not found: {missing[:10]}")

    df['ensemble_uncertainty'] = ensemble_uncertainty
    df_clean = df.dropna(subset=['ensemble_uncertainty'])

    y = df_clean['is_correct'].values.astype(int)
    # higher variance = more uncertain = more likely misclassified, so invert for AUROC scoring
    scores = -df_clean['ensemble_uncertainty'].values
    ensemble_auroc = roc_auc_score(y, scores)

    print(f"\nEnsemble members trained: {N_ENSEMBLE}")
    print(f"Melanoma cases evaluated: n = {len(df_clean)}")
    print(f"Ensemble variance AUROC: {ensemble_auroc:.3f}")
    print(f"  (compare against: ECS AUROC 0.531, entropy AUROC 0.522, PLEU AUROC 0.523)")

    with open(OUTPUT_TXT, 'w') as f:
        f.write(f"n_ensemble_members: {N_ENSEMBLE}\n")
        f.write(f"n_melanoma_evaluated: {len(df_clean)}\n")
        f.write(f"ensemble_variance_auroc: {ensemble_auroc:.3f}\n")
        f.write(f"training_time_per_model_minutes: {[t/60 for t in timings]}\n")

    print(f"\nSaved results to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()