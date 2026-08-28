import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.load_checkpoint import load_model
from src.data.dataloader import get_transforms, SkinLesionDataset
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model("results/20260822_224442_baseline_resnet50/best_model.pth").to(device).eval()

# Load test split
splits = pd.read_csv("data/splits/ham10000_hospital_splits.csv")
test_df = splits[splits['split'] == 'test']

test_transform = get_transforms('val')
test_dataset = SkinLesionDataset(test_df, img_dir="data/raw/ham10000/images/", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
n_classes = len(classes)
cm = np.zeros((n_classes, n_classes), dtype=int)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        for p, l in zip(preds, labels):
            cm[l, p] += 1

# Normalize
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Normalized Frequency'})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix — Vienna External Test Set')
plt.tight_layout()
plt.savefig("results/fig_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved to results/fig_confusion_matrix.png")