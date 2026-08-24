import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

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
    
    # Save
    save_dir = os.path.dirname(checkpoint_path)
    import json
    with open(os.path.join(save_dir, 'vienna_external_metrics_fixed.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    confidences = probs.max(axis=1)
    plot_reliability_diagram(confidences, preds, labels, os.path.join(save_dir, 'vienna_reliability_diagram_fixed.png'))
    
    print(f"\nSaved to {save_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)