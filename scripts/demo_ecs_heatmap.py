import torch
import sys
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.explainability.pleu import PatchLevelEpistemicUncertainty
from src.explainability.ecs import ExplanationConsistencyScore
from src.explainability.trust_heatmap import TrustHeatmap
from scripts.load_checkpoint import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_PATH = "results/20260822_224442_baseline_resnet50/best_model.pth"
model = load_model(CHECKPOINT_PATH).to(device)

test_cases = [
    ("data/raw/ham10000/images/ISIC_0029913.jpg", "correct_high_ecs"),
    ("data/raw/ham10000/images/ISIC_0028082.jpg", "correct_low_ecs"),
    ("data/raw/ham10000/images/ISIC_0030281.jpg", "wrong_low_ecs"),
]

pleu = PatchLevelEpistemicUncertainty(model, patch_size=32, stride=16)
ecs = ExplanationConsistencyScore(model, model.backbone.layer4, T=10)
heatmap = TrustHeatmap()

os.makedirs("results/trust_heatmaps", exist_ok=True)

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for img_path, tag in test_cases:
    if not os.path.exists(img_path):
        print(f"SKIP: {img_path} not found")
        continue
    
    pil_img = Image.open(img_path).convert('RGB')
    img_tensor = val_transform(pil_img).to(device)
    
    pleu_result = pleu(img_tensor)
    patch_unc = pleu_result['patch_uncertainties'].cpu().numpy()
    patch_pos = pleu_result['patch_positions']
    
    h, w = img_tensor.shape[1], img_tensor.shape[2]
    pleu_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    patch_size = pleu.patch_size
    
    for (y, x), unc in zip(patch_pos, patch_unc):
        y_end = min(y + patch_size, h)
        x_end = min(x + patch_size, w)
        pleu_map[y:y_end, x:x_end] += unc
        count_map[y:y_end, x:x_end] += 1
    
    count_map[count_map == 0] = 1
    pleu_map = pleu_map / count_map
    
    ecs_result = ecs.compute(pil_img)
    ecs_score = ecs_result['ecs_score']
    
    img_display = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)
    
    trust, fig = heatmap.generate(img_display, pleu_map, ecs_score=ecs_score)
    
    out_path = f"results/trust_heatmaps/{tag}_ecs{ecs_score:.3f}.png"
    heatmap.save(fig, out_path)
    print(f"DONE: {tag} | ECS={ecs_score:.3f} | PLEU_reject={pleu_result['should_reject']}")

print("\nAll figures saved to results/trust_heatmaps/")