import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.baselines import SkinLesionClassifier
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as sk_auroc
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline          # ADD THIS
from sklearn.preprocessing import StandardScaler  

# ==================== CONFIG ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"results\20260822_224442_baseline_resnet50\best_model.pth"  # <-- your model path
CSV_PATH = "results/ecs_melanoma_results_with_stats_v2.csv"  # <-- your ECS results
HAM_IMG_DIR = "data/raw/ham10000/images"  # <-- HAM10000 images
NEAR_OOD_DIR = "data/raw/isic_near_ood"   # <-- ISIC seborrheic keratosis + solar lentigo images
OUTPUT_TXT = "results/ood_trust_evaluation.txt"


# ==================== PATH VALIDATION ====================
def validate_paths():
    """Fail loudly and clearly, before doing any real work, if inputs are missing."""
    problems = []
    if not os.path.isfile(MODEL_PATH):
        problems.append(f"MODEL_PATH does not exist: '{MODEL_PATH}'")
    if not os.path.isfile(CSV_PATH):
        problems.append(f"CSV_PATH does not exist: '{CSV_PATH}'")
    if not os.path.isdir(HAM_IMG_DIR):
        problems.append(f"HAM_IMG_DIR does not exist: '{HAM_IMG_DIR}'")
    if not os.path.isdir(NEAR_OOD_DIR):
        problems.append(f"NEAR_OOD_DIR does not exist: '{NEAR_OOD_DIR}'")

    out_dir = os.path.dirname(OUTPUT_TXT)
    if out_dir and not os.path.isdir(out_dir):
        problems.append(f"Output directory does not exist: '{out_dir}'")

    if problems:
        msg = "Cannot run — fix these paths at the top of the script before rerunning:\n"
        msg += "\n".join(f"  - {p}" for p in problems)
        raise FileNotFoundError(msg)


# ==================== LOAD MODEL ====================
def load_model():
    model = SkinLesionClassifier(
        model_name='resnet50',
        num_classes=7,
        pretrained=False,  # we're loading trained weights, not ImageNet init
        dropout=0.5
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from full training checkpoint (epoch={checkpoint.get('epoch', '?')}, "
              f"val_acc={checkpoint.get('val_acc', '?')})")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()
    return model


# ==================== TRANSFORM ====================
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ==================== HELPERS ====================
def get_prediction(img_path, model):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()[0]


def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-8))


# ==================== 1. NEAR-OOD EVALUATION ====================
def evaluate_near_ood(model):
    near_ood_files = []
    for root, _, files in os.walk(NEAR_OOD_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                near_ood_files.append(os.path.join(root, f))

    if len(near_ood_files) == 0:
        raise RuntimeError(
            f"No near-OOD images found under '{NEAR_OOD_DIR}'. "
            "Check the folder actually contains .jpg/.jpeg/.png files "
            "before trusting any AUROC computed here."
        )

    df = pd.read_csv(CSV_PATH)
    if 'image_id' not in df.columns:
        raise ValueError(f"CSV_PATH '{CSV_PATH}' has no 'image_id' column. Columns found: {df.columns.tolist()}")

    in_dist_ids = df['image_id'].tolist()

    in_dist_scores = []
    missing_in_dist = 0
    for img_id in in_dist_ids:
        img_path = os.path.join(HAM_IMG_DIR, f"{img_id}.jpg")
        if os.path.exists(img_path):
            probs = get_prediction(img_path, model)
            in_dist_scores.append(np.max(probs))
        else:
            missing_in_dist += 1

    if missing_in_dist > 0:
        print(f"WARNING: {missing_in_dist}/{len(in_dist_ids)} in-distribution images listed in the CSV "
              f"were not found under '{HAM_IMG_DIR}'. AUROC below is computed only on the "
              f"{len(in_dist_scores)} images that were actually found — check this isn't hiding a path bug.")

    if len(in_dist_scores) == 0:
        raise RuntimeError(
            "No in-distribution images could be loaded — every image_id in the CSV was missing from "
            f"HAM_IMG_DIR ('{HAM_IMG_DIR}'). AUROC cannot be computed. Check the path and file extension."
        )

    near_ood_scores = []
    for img_path in near_ood_files:
        probs = get_prediction(img_path, model)
        near_ood_scores.append(np.max(probs))

    # AUROC: in-dist = 1, near-OOD = 0. Lower MSP = more OOD
    y_true = [1] * len(in_dist_scores) + [0] * len(near_ood_scores)
    y_score = in_dist_scores + near_ood_scores

    auroc = sk_auroc(y_true, y_score)

    return {
        'near_ood_n': len(near_ood_files),
        'in_dist_n': len(in_dist_scores),
        'auroc': auroc,
        'mean_msp_in': np.mean(in_dist_scores),
        'mean_msp_ood': np.mean(near_ood_scores)
    }


# ==================== 2. LEARNED TRUST SCORE AUROC ====================
def evaluate_learned_trust(model):
    df = pd.read_csv(CSV_PATH)

    # Features: [PLEU, ECS, confidence] -> predict is_correct
    required = ['pleu', 'ecs_score', 'confidence', 'is_correct']
    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        # Previously this silently filled missing columns with constants (pleu=0.5, ecs_score=0.8),
        # which produces a fake ~0.5 AUROC for any missing feature and would have been reported
        # as a real result. That is not safe to do automatically — stop instead.
        raise ValueError(
            f"CSV_PATH '{CSV_PATH}' is missing required column(s): {missing_cols}. "
            f"Columns found: {df.columns.tolist()}.\n"
            "Refusing to auto-fill these with placeholder constants (e.g. pleu=0.5) because a constant "
            "feature carries zero information and would silently produce a meaningless ~0.5 AUROC that "
            "looks like a real result. Compute the missing column(s) properly (e.g. from your PLEU "
            "script) and merge them into this CSV on 'image_id' before rerunning."
        )

    # Optional sanity check: a column that is constant (zero variance) is as good as missing.
    for col in ['pleu', 'ecs_score', 'confidence']:
        if df[col].nunique(dropna=True) <= 1:
            raise ValueError(
                f"Column '{col}' in '{CSV_PATH}' has no variance (all values identical: "
                f"{df[col].iloc[0]!r}). This looks like a placeholder/dummy column, not real data. "
                "Fix the source of this column before trusting any AUROC computed from it."
            )

    X = df[['pleu', 'ecs_score', 'confidence']].values
    y = df['is_correct'].values.astype(int)

    # 5-fold cross-validated out-of-sample predictions (more robust than a single split)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    preds = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:, 1]

    trust_auroc = sk_auroc(y, preds)

    # Individual AUROCs for comparison
    pleu_auroc = sk_auroc(y, -df['pleu'])       # higher PLEU = more error, so invert
    ecs_auroc = sk_auroc(y, df['ecs_score'])    # higher ECS = more likely correct (per Table IV)
    conf_auroc = sk_auroc(y, df['confidence'])  # higher conf = more correct

    return {
        'trust_auroc': trust_auroc,
        'pleu_auroc': pleu_auroc,
        'ecs_auroc': ecs_auroc,
        'conf_auroc': conf_auroc,
        'n_samples': len(df)
    }


# ==================== MAIN ====================
if __name__ == "__main__":
    print("Validating paths...")
    validate_paths()

    print("Loading model...")
    model = load_model()

    print("\n" + "=" * 60)
    print("1. NEAR-OOD EVALUATION")
    print("=" * 60)
    near_ood_results = evaluate_near_ood(model)
    print(f"Near-OOD images evaluated: n = {near_ood_results['near_ood_n']}")
    print(f"In-distribution images: n = {near_ood_results['in_dist_n']}")
    print(f"AUROC (MSP): {near_ood_results['auroc']:.3f}")
    print(f"Mean MSP (in-dist): {near_ood_results['mean_msp_in']:.3f}")
    print(f"Mean MSP (near-OOD): {near_ood_results['mean_msp_ood']:.3f}")

    print("\n" + "=" * 60)
    print("2. LEARNED TRUST SCORE DISCRIMINATION")
    print("=" * 60)
    trust_results = evaluate_learned_trust(model)
    print(f"Samples: n = {trust_results['n_samples']}")
    print(f"Learned Trust AUROC: {trust_results['trust_auroc']:.3f}")
    print(f"PLEU AUROC: {trust_results['pleu_auroc']:.3f}")
    print(f"ECS AUROC: {trust_results['ecs_auroc']:.3f}")
    print(f"Confidence AUROC: {trust_results['conf_auroc']:.3f}")

    # Save to file
    with open(OUTPUT_TXT, 'w') as f:
        f.write(f"near_ood_n: {near_ood_results['near_ood_n']}\n")
        f.write(f"near_ood_auroc: {near_ood_results['auroc']:.3f}\n")
        f.write(f"learned_trust_auroc: {trust_results['trust_auroc']:.3f}\n")
        f.write(f"pleu_auroc: {trust_results['pleu_auroc']:.3f}\n")
        f.write(f"ecs_auroc: {trust_results['ecs_auroc']:.3f}\n")
        f.write(f"conf_auroc: {trust_results['conf_auroc']:.3f}\n")

    print(f"\nSaved results to: {OUTPUT_TXT}")