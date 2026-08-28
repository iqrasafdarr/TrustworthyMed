import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from src.models.baselines import build_model
from src.data.dataloader import SkinLesionDataset, get_transforms
from src.evaluation.calibration import evaluate_model, plot_reliability_diagram

class FixedClassDataset(SkinLesionDataset):
    """
    Uses a fixed class mapping instead of rebuilding from dataframe.
    This prevents label mismatch between training and external test.
    """
    def __init__(self, df, img_dir, class_names, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')

        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, row['image_id'] + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, row['image_id'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row['dx']]
        return image, label, row['image_id']

from PIL import Image

def main(checkpoint_path):
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')

    # FIXED CLASS ORDER (same as training)
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    # Load Vienna test
    splits = pd.read_csv('data/splits/ham10000_hospital_splits.csv')
    test_df = splits[splits['split'] == 'test']

    print(f"External Validation (Vienna Hospital): {len(test_df)} images")
    print(f"Using fixed class mapping: {CLASS_NAMES}")

    if len(test_df) == 0:
        print("ERROR: No test images!")
        return

    # Use fixed class dataset
    test_ds = FixedClassDataset(test_df, config['data']['ham10000_images'], CLASS_NAMES, get_transforms('val'))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # Load model
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Evaluate
    metrics, probs, preds, labels, ids = evaluate_model(model, test_loader, device, num_classes=7)

    print(f"\n{'='*60}")
    print("EXTERNAL VALIDATION RESULTS (Vienna Hospital)")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1: {metrics['f1']*100:.2f}%")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")

    # Save metrics
    save_dir = os.path.dirname(checkpoint_path)
    import json
    with open(os.path.join(save_dir, 'vienna_external_metrics_fixed.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    confidences = probs.max(axis=1)
    plot_reliability_diagram(confidences, preds, labels, os.path.join(save_dir, 'vienna_reliability_diagram_fixed.png'))

    # ============================================================
    # CONFUSION MATRIX (REAL — from actual predictions)
    # ============================================================
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Normalized Frequency'},
                linewidths=0.5, linecolor='white',
                annot_kws={"size": 11, "weight": "bold"},
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Normalized Confusion Matrix — Vienna External Test Set ($N=439$)', 
                 fontsize=14, fontweight='bold', pad=15)

    # Highlight melanoma row/column
    ax.add_patch(plt.Rectangle((4, 4), 1, 1, fill=False, edgecolor='red', lw=3))
    ax.add_patch(plt.Rectangle((5, 4), 1, 1, fill=False, edgecolor='darkorange', lw=2, linestyle='--'))

    # Annotate melanoma accuracy
    mel_acc = cm[4, 4] / cm[4, :].sum() * 100
    ax.text(4.5, 4.5, f'{mel_acc:.1f}%', ha='center', va='center', 
            fontsize=12, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))

    plt.tight_layout()

    # Save in TWO places: checkpoint dir + results dir for paper
    cm_path1 = os.path.join(save_dir, 'fig_confusion_matrix.png')
    cm_path2 = 'results/fig_confusion_matrix.png'

    fig.savefig(cm_path1, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(cm_path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\n✅ Confusion matrix saved to:")
    print(f"   {cm_path1}")
    print(f"   {cm_path2}")
    print(f"\nMelanoma accuracy from CM: {mel_acc:.2f}%")
    print(f"Nevus accuracy from CM: {cm[5,5]/cm[5,:].sum()*100:.2f}%")
    print(f"\nSaved all results to {save_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)