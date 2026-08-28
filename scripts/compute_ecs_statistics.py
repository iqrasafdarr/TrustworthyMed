import torch
import pandas as pd
import numpy as np
from PIL import Image
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import sys

sys.path.insert(0, 'scripts')
from load_checkpoint import SkinLesionClassifier

# ---- CONFIG ----
RESULTS_CSV = "results/ecs_melanoma_results.csv"
CHECKPOINT_PATH = "results/20260822_224442_baseline_resnet50/best_model.pth"
IMAGE_DIR = "data/raw/ham10000/images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- LOAD ECS RESULTS ----
df = pd.read_csv(RESULTS_CSV)
correct = df[df['is_correct'] == True]['ecs_score']
incorrect = df[df['is_correct'] == False]['ecs_score']

print("=" * 60)
print("ECS STATISTICAL ANALYSIS")
print("=" * 60)
print(f"Correct melanomas:     n={len(correct)}, mean={correct.mean():.4f}, std={correct.std():.4f}")
print(f"Misclassified melanomas: n={len(incorrect)}, mean={incorrect.mean():.4f}, std={incorrect.std():.4f}")
print(f"Mean difference: {correct.mean() - incorrect.mean():.4f}")

# ---- 1. T-TEST ----
t_stat, t_p = ttest_ind(correct, incorrect, equal_var=False)
print(f"\nWelch's t-test: t = {t_stat:.4f}, p = {t_p:.4f}")

# ---- 2. MANN-WHITNEY U (more robust for skewed data) ----
u_stat, u_p = mannwhitneyu(correct, incorrect, alternative='two-sided')
print(f"Mann-Whitney U: U = {u_stat:.2f}, p = {u_p:.4f}")

# ---- 3. COHEN'S D (effect size) ----
pooled_var = ((len(correct)-1)*correct.var() + (len(incorrect)-1)*incorrect.var()) / (len(correct)+len(incorrect)-2)
cohens_d = (correct.mean() - incorrect.mean()) / (pooled_var ** 0.5)
print(f"Cohen's d: {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print("  → Negligible effect size")
elif abs(cohens_d) < 0.5:
    print("  → Small effect size")
elif abs(cohens_d) < 0.8:
    print("  → Medium effect size")
else:
    print("  → Large effect size")

# ---- 4. TAIL ANALYSIS (where the signal actually lives) ----
print("\n" + "=" * 60)
print("TAIL BEHAVIOR (Threshold-based)")
print("=" * 60)

for tau in [0.90, 0.95, 0.99]:
    flagged = df[df['ecs_score'] < tau]
    if len(flagged) > 0:
        catch_rate = (~flagged['is_correct']).sum() / (~df['is_correct']).sum() * 100
        false_alarm = (flagged['is_correct']).sum() / len(flagged) * 100
        flagged_pct = len(flagged) / len(df) * 100
        print(f"τ = {tau}: Flags {flagged_pct:.1f}% of cases | Catches {catch_rate:.1f}% of errors | {false_alarm:.1f}% false alarm")

# ---- 5. DECILE ANALYSIS ----
print("\n" + "=" * 60)
print("DECILE ANALYSIS")
print("=" * 60)
df['ecs_decile'] = pd.qcut(df['ecs_score'], 10, labels=False, duplicates='drop')
for decile in sorted(df['ecs_decile'].unique()):
    subset = df[df['ecs_decile'] == decile]
    err_rate = (~subset['is_correct']).sum() / len(subset) * 100
    print(f"Decile {decile}: {len(subset)} cases, error rate = {err_rate:.1f}%")

# ---- 6. ENTROPY BASELINE COMPARISON ----
print("\n" + "=" * 60)
print("ENTROPY BASELINE COMPARISON")
print("=" * 60)

def load_model():
    model = SkinLesionClassifier(num_classes=7)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

entropies = []
print("Computing entropy for each image...")

for _, row in df.iterrows():
    img_path = f"{IMAGE_DIR}/{row['image_id']}.jpg"
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        try:
            pil_img = Image.open(img_path.replace('.jpg', '.png')).convert("RGB")
        except FileNotFoundError:
            entropies.append(np.nan)
            continue
    
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    
    # Entropy = -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
    entropies.append(entropy)

df['entropy'] = entropies
df = df.dropna(subset=['entropy'])

# AUROC: Can this signal detect misclassification?
# For ECS: lower score = more error → negate
ecs_auroc = roc_auc_score(~df['is_correct'], -df['ecs_score'])
# For entropy: higher entropy = more error
ent_auroc = roc_auc_score(~df['is_correct'], df['entropy'])

print(f"\nAUROC for detecting misclassification:")
print(f"  ECS (lower = worse):     {ecs_auroc:.3f}")
print(f"  Entropy (higher = worse): {ent_auroc:.3f}")

if ecs_auroc > ent_auroc:
    print(f"  → ECS outperforms entropy by {(ecs_auroc - ent_auroc)*100:.1f} percentage points")
else:
    print(f"  → Entropy outperforms ECS by {(ent_auroc - ecs_auroc)*100:.1f} percentage points")
    print("  → Frame ECS as COMPLEMENTARY to entropy, not a replacement")

# ---- 7. SAVE ENHANCED CSV ----
df.to_csv("results/ecs_melanoma_results_with_stats.csv", index=False)
print("\nSaved enhanced results to: results/ecs_melanoma_results_with_stats.csv")
print("=" * 60)