"""
Compute real PLEU and confidence scores for the melanoma cases in
results/ecs_melanoma_results_with_stats.csv, and merge them in as new columns.

This fills the gap flagged by evaluate_ood_and_trust.py: that CSV had
ecs_score/entropy/is_correct but no real 'pleu' or 'confidence' columns,
so the learned trust score (PLEU + ECS + confidence) couldn't be computed.

Run from your project root:
    python -m scripts.compute_pleu_confidence
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # so `src` is importable

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.baselines import SkinLesionClassifier
from src.explainability.pleu import PatchLevelEpistemicUncertainty

# ==================== CONFIG ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"results\20260822_224442_baseline_resnet50\best_model.pth"
CSV_PATH = "results/ecs_melanoma_results_with_stats.csv"
HAM_IMG_DIR = "data/raw/ham10000/images"
OUTPUT_CSV_PATH = "results/ecs_melanoma_results_with_stats_v2.csv"  # new file, doesn't overwrite original

# PLEU hyperparameters — match whatever your paper reports (patch_size=32, stride=16 per the draft)
PLEU_PATCH_SIZE = 32
PLEU_STRIDE = 16
# NOTE: calibrated from diagnose_pleu_scale.py output on 2026-08-28. Real patch-uncertainty
# values ranged 0.00-0.125 (mean 0.017), so the old default of 0.3 flagged zero patches ever,
# producing pleu=0.0 for every image. 0.025 sits near the 75th percentile of the observed
# distribution, so roughly the most uncertain ~25% of patches get flagged per image on average.
PLEU_UNCERTAINTY_THRESHOLD = 0.025
PLEU_REJECT_FRACTION = 0.4
PLEU_MC_ITERATIONS = 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model():
    model = SkinLesionClassifier(model_name='resnet50', num_classes=7, pretrained=False, dropout=0.5)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint (epoch={checkpoint.get('epoch', '?')}, "
              f"val_acc={checkpoint.get('val_acc', '?')})")
    else:
        model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    return model


def compute_confidence(model, image_tensor):
    """Max softmax probability, dropout OFF (proper eval-mode confidence)."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(DEVICE))
        probs = F.softmax(logits, dim=1)
    return probs.max().item()


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")
    if not os.path.isdir(HAM_IMG_DIR):
        raise FileNotFoundError(f"HAM_IMG_DIR not found: {HAM_IMG_DIR}")

    print("Loading model...")
    model = load_model()

    pleu_computer = PatchLevelEpistemicUncertainty(
        model,
        patch_size=PLEU_PATCH_SIZE,
        stride=PLEU_STRIDE,
        uncertainty_threshold=PLEU_UNCERTAINTY_THRESHOLD,
        reject_fraction=PLEU_REJECT_FRACTION,
    )

    df = pd.read_csv(CSV_PATH)
    if 'image_id' not in df.columns:
        raise ValueError(f"CSV_PATH has no 'image_id' column. Columns found: {df.columns.tolist()}")

    print(f"Computing PLEU + confidence for {len(df)} images...")
    pleu_scores = []
    confidences = []
    missing = []

    for i, row in df.iterrows():
        img_id = row['image_id']
        img_path = os.path.join(HAM_IMG_DIR, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            missing.append(img_id)
            pleu_scores.append(None)
            confidences.append(None)
            continue

        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)  # (C, H, W), CPU tensor — PLEU extracts patches from this directly

        # Confidence: eval mode, dropout off
        conf = compute_confidence(model, tensor)

        # PLEU: this internally sets model.train() to keep dropout active for MC sampling.
        # Reset to eval() immediately after so nothing downstream is silently affected.
        pleu_tensor = tensor.to(DEVICE)
        result = pleu_computer(pleu_tensor)
        model.eval()

        pleu_scores.append(result['pleu_score'])
        confidences.append(conf)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df)} done...")

    if missing:
        print(f"\nWARNING: {len(missing)} images not found under '{HAM_IMG_DIR}': {missing[:10]}"
              f"{' ...' if len(missing) > 10 else ''}")
        print("Rows for these images will have pleu/confidence = None — check before using this CSV downstream.")

    df['pleu'] = pleu_scores
    df['confidence'] = confidences

    n_before = len(df)
    df_clean = df.dropna(subset=['pleu', 'confidence'])
    n_after = len(df_clean)
    if n_after < n_before:
        print(f"\nDropped {n_before - n_after} rows with missing pleu/confidence "
              f"(images not found) before saving.")

    df_clean.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved: {OUTPUT_CSV_PATH}  ({n_after} rows, columns: {df_clean.columns.tolist()})")
    print(f"\nSanity check — pleu stats: mean={df_clean['pleu'].mean():.3f}, "
          f"std={df_clean['pleu'].std():.3f}, unique values={df_clean['pleu'].nunique()}")
    print(f"Sanity check — confidence stats: mean={df_clean['confidence'].mean():.3f}, "
          f"std={df_clean['confidence'].std():.3f}, unique values={df_clean['confidence'].nunique()}")

    if df_clean['pleu'].nunique() <= 1:
        print("\nWARNING: pleu has no variance — something is still wrong (all patches uncertain/certain "
              "identically). Do not use this column downstream yet.")

    print(f"\nNext step: once this looks right, update CSV_PATH in evaluate_ood_and_trust.py to point at "
          f"'{OUTPUT_CSV_PATH}' (or rename/overwrite the original) and rerun Part 2.")


if __name__ == "__main__":
    main()