import torch
import pandas as pd
from PIL import Image
import sys
import csv

sys.path.insert(0, 'scripts')
from load_checkpoint import SkinLesionClassifier
from src.explainability.ecs import ExplanationConsistencyScore

# ---- CONFIG ----
CHECKPOINT_PATH = "results/20260822_224442_baseline_resnet50/best_model.pth"
IMAGE_DIR = "data/raw/ham10000/images"
SPLITS_CSV = "data/splits/ham10000_hospital_splits.csv"
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
MELANOMA_CLASS_IDX = CLASS_NAMES.index("mel")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = SkinLesionClassifier(num_classes=7)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    return model


def main():
    model = load_model()
    target_layer = model.backbone.layer4[-1]
    ecs_module = ExplanationConsistencyScore(model, target_layer, T=10, reject_threshold=0.5)

    # Load VAL split only, filter melanoma
    splits_df = pd.read_csv(SPLITS_CSV)
    val_df = splits_df[splits_df['split'] == 'val']
    melanoma_rows = val_df[val_df['dx'] == 'mel']
    
    print(f"Found {len(melanoma_rows)} melanoma images in val split.")
    print("Running ECS (this may take a few minutes)...")

    correct_scores = []
    incorrect_scores = []
    results = []

    for i, (_, row) in enumerate(melanoma_rows.iterrows()):
        img_path = f"{IMAGE_DIR}/{row['image_id']}.jpg"
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            try:
                pil_img = Image.open(img_path.replace('.jpg', '.png')).convert("RGB")
            except FileNotFoundError:
                continue

        # Prediction (no transform needed - ECS handles it internally)
        model.eval()
        with torch.no_grad():
            from torchvision import transforms
            t = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = t(pil_img).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            predicted_class = logits.argmax(dim=1).item()
        
        is_correct = (predicted_class == MELANOMA_CLASS_IDX)

        # ECS - pass PIL image directly
        result = ecs_module.compute(pil_img)
        score = result["ecs_score"]

        results.append({
            "image_id": row['image_id'],
            "predicted": CLASS_NAMES[predicted_class],
            "true": "mel",
            "is_correct": is_correct,
            "ecs_score": score
        })

        if is_correct:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(melanoma_rows)}...")

    # Save CSV
    import os
    os.makedirs('results', exist_ok=True)
    with open("results/ecs_melanoma_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "predicted", "true", "is_correct", "ecs_score"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "="*50)
    print(f"Correctly classified: {len(correct_scores)}")
    if correct_scores:
        print(f"  Average ECS: {sum(correct_scores)/len(correct_scores):.3f}")
    print(f"Misclassified: {len(incorrect_scores)}")
    if incorrect_scores:
        print(f"  Average ECS: {sum(incorrect_scores)/len(incorrect_scores):.3f}")
    print("Results saved to: results/ecs_melanoma_results.csv")
    print("="*50)


if __name__ == "__main__":
    main()