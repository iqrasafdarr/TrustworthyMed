"""
Diagnostic: inspect raw PLEU patch-uncertainty values (before thresholding)
for a handful of melanoma images, so we can pick a sensible uncertainty_threshold
instead of guessing. The default threshold=0.3 produced pleu=0.0 for all 150
images, meaning it's likely too high for the actual scale of these values.

Run from your project root:
    python -m scripts.diagnose_pleu_scale
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.baselines import SkinLesionClassifier
from src.explainability.pleu import PatchLevelEpistemicUncertainty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"results\20260822_224442_baseline_resnet50\best_model.pth"
CSV_PATH = "results/ecs_melanoma_results_with_stats.csv"
HAM_IMG_DIR = "data/raw/ham10000/images"
N_SAMPLE_IMAGES = 10  # just enough to see the distribution shape

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
    else:
        model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    return model


def main():
    print("Loading model...")
    model = load_model()

    # Use a low threshold just to get __call__ to run; we only care about the raw
    # 'patch_uncertainties' tensor it returns, not the thresholded pleu_score itself.
    pleu_computer = PatchLevelEpistemicUncertainty(
        model, patch_size=32, stride=16, uncertainty_threshold=0.0, reject_fraction=0.4
    )

    df = pd.read_csv(CSV_PATH).head(N_SAMPLE_IMAGES)

    all_uncertainties = []
    for i, row in df.iterrows():
        img_id = row['image_id']
        img_path = os.path.join(HAM_IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            print(f"  skipping {img_id}, image not found")
            continue

        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).to(DEVICE)

        result = pleu_computer(tensor)
        model.eval()  # reset after MC dropout pass

        u = result['patch_uncertainties']  # tensor of per-patch uncertainty values
        u_np = u.detach().cpu().numpy()
        all_uncertainties.extend(u_np.tolist())

        print(f"{img_id}: n_patches={len(u_np)}, min={u_np.min():.5f}, max={u_np.max():.5f}, "
              f"mean={u_np.mean():.5f}, std={u_np.std():.5f}")

    import numpy as np
    all_uncertainties = np.array(all_uncertainties)
    print(f"\n{'='*60}")
    print(f"OVERALL across {len(all_uncertainties)} patches from {N_SAMPLE_IMAGES} images:")
    print(f"  min={all_uncertainties.min():.5f}")
    print(f"  max={all_uncertainties.max():.5f}")
    print(f"  mean={all_uncertainties.mean():.5f}")
    print(f"  std={all_uncertainties.std():.5f}")
    print(f"  50th percentile (median): {np.percentile(all_uncertainties, 50):.5f}")
    print(f"  75th percentile: {np.percentile(all_uncertainties, 75):.5f}")
    print(f"  90th percentile: {np.percentile(all_uncertainties, 90):.5f}")
    print(f"  95th percentile: {np.percentile(all_uncertainties, 95):.5f}")
    print(f"  99th percentile: {np.percentile(all_uncertainties, 99):.5f}")
    print(f"{'='*60}")
    print(f"\nYour current threshold (0.3) is likely far above max={all_uncertainties.max():.5f}.")
    print(f"A reasonable threshold would sit around the 75th-90th percentile shown above, "
          f"so roughly 10-25% of patches get flagged per image on average — giving pleu_score "
          f"real variance instead of a constant 0.")


if __name__ == "__main__":
    main()