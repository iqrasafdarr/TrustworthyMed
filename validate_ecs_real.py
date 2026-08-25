import torch
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import os
import json
from src.explainability.ecs import ExplanationConsistencyScore

# --- Config ---
CHECKPOINT = "results/20260822_224442_baseline_resnet50/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 10

# --- Load model ---
model = resnet50(num_classes=7)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.train()  # dropout ON for ECS

ecs = ExplanationConsistencyScore(model, target_layer=model.layer4[-1], T=T)

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Run on a few Vienna images ---
# Replace with actual paths to your Vienna validation images
test_images = [
    # "data/vienna/val/...",
]

results = []
for img_path in test_images[:5]:  # start with 5
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    out = ecs.compute(tensor)
    results.append({"image": img_path, **out})
    print(f"{img_path}: ECS={out['ecs_score']:.3f}, reject={out['should_reject']}, class={out['fixed_class']}")

# Save
with open("results/ecs_validation.json", "w") as f:
    json.dump(results, f, indent=2)