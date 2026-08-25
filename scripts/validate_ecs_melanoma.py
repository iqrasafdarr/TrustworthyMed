import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

from src.explainability.ecs import ExplanationConsistencyScore

# ---- CONFIG: adjust these paths to match your repo ----
CHECKPOINT_PATH = "results/20260822_224442_baseline_resnet50/best_model.pth"
METADATA_CSV = "data/raw/ham10000/HAM10000_metadata.csv"
IMAGE_DIR = "data/raw/ham10000/images"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # standard HAM10000 order
MELANOMA_CLASS_IDX = CLASS_NAMES.index("mel")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)  # handle either format
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    return model


def get_prediction(model, image_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        return logits.argmax(dim=1).item()


def main():
    model = load_model()
    target_layer = model.layer4[-1]  # last conv block of ResNet-50
    ecs_module = ExplanationConsistencyScore(model, target_layer, T=10, reject_threshold=0.5)

    metadata = pd.read_csv(METADATA_CSV)
    melanoma_rows = metadata[metadata["dx"] == "mel"]

    correct_scores = []
    incorrect_scores = []

    for _, row in melanoma_rows.iterrows():
        img_path = f"{IMAGE_DIR}/{row['image_id']}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            continue

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        predicted_class = get_prediction(model, image_tensor)
        true_label_correct = (predicted_class == MELANOMA_CLASS_IDX)

        result = ecs_module.compute(image_tensor)
        score = result["ecs_score"]

        if true_label_correct:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)

    print(f"Melanoma images correctly classified: {len(correct_scores)}")
    print(f"  Average ECS: {sum(correct_scores)/len(correct_scores):.3f}" if correct_scores else "  N/A")
    print(f"Melanoma images misclassified: {len(incorrect_scores)}")
    print(f"  Average ECS: {sum(incorrect_scores)/len(incorrect_scores):.3f}" if incorrect_scores else "  N/A")


if __name__ == "__main__":
    main()