import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

from src.explainability.pleu import PatchLevelEpistemicUncertainty
from src.explainability.ecs import ExplanationConsistencyScore

# ---- CONFIG: fill these in once you have real paths ----
CHECKPOINT_PATH = "results/20260822_224442_baseline_resnet50/best_model.pth"
METADATA_CSV = "data/processed/ham10000_splits.csv"   # adjust once confirmed
IMAGE_DIR = "data/raw/ham10000/images"                 # adjust once confirmed
OUTPUT_CSV = "results/trust_validation_results.csv"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MELANOMA_IDX = CLASS_NAMES.index("mel")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_IMAGES = None  # set an int (e.g. 100) to test on a subset first

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(DEVICE)
    return model


def entropy(probs):
    """Standard entropy, used as the mc_uncertainty scalar (normalized to [0,1])."""
    probs = np.clip(probs, 1e-8, 1.0)
    h = -np.sum(probs * np.log(probs))
    return h / np.log(len(probs))  # normalize by max possible entropy


def main():
    model = load_model()
    target_layer = model.layer4[-1]

    pleu = PatchLevelEpistemicUncertainty(model)
    ecs_module = ExplanationConsistencyScore(model, target_layer, T=10, reject_threshold=0.5)

    metadata = pd.read_csv(METADATA_CSV)
    if N_IMAGES:
        metadata = metadata.head(N_IMAGES)

    rows = []
    for i, row in metadata.iterrows():
        img_path = Path(IMAGE_DIR) / f"{row['image_id']}.jpg"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        # --- confidence + prediction (dropout OFF for a clean point estimate) ---
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(probs.argmax())
        confidence = float(probs.max())

        # --- ECS (also gives us the fixed_class used for explanation) ---
        ecs_result = ecs_module.compute(tensor)

        # --- MC uncertainty via entropy of mean probs across T passes ---
        mc_uncertainty = entropy(probs)  # simple version; swap for PLEU's patch-based score if preferred

        true_label = row.get("dx", None)
        true_idx = CLASS_NAMES.index(true_label) if true_label in CLASS_NAMES else None

        rows.append({
            "image_id": row["image_id"],
            "true_label": true_label,
            "true_idx": true_idx,
            "pred_class": pred_class,
            "confidence": confidence,
            "mc_uncertainty": mc_uncertainty,
            "ecs_score": ecs_result["ecs_score"],
            "should_reject_ecs": ecs_result["should_reject"],
            "is_correct": (pred_class == true_idx),
            "is_melanoma_true": (true_idx == MELANOMA_IDX),
            "is_melanoma_pred": (pred_class == MELANOMA_IDX),
        })

        if i % 20 == 0:
            print(f"Processed {i} images...")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} results to {OUTPUT_CSV}")

    # Quick summary printed immediately
    melanoma_df = df[df["is_melanoma_true"]]
    print("\n--- Melanoma cases ---")
    print(f"Correct: {melanoma_df[melanoma_df['is_correct']]['ecs_score'].mean():.3f} avg ECS")
    print(f"Incorrect: {melanoma_df[~melanoma_df['is_correct']]['ecs_score'].mean():.3f} avg ECS")


if __name__ == "__main__":
    main()