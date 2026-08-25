import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image

from src.models.baselines import build_model
from src.data.dataloader import get_transforms
from src.robustness.clinical_corruptions import ClinicalCorruptions

def evaluate_on_corruption(model, device, test_df, img_dir, corruption_type, severity, class_to_idx):
    model.eval()
    correct = 0
    total = 0
    transform = get_transforms('val')
    
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            img_id = row['image_id']
            true_label = row['dx']
            
            if true_label not in class_to_idx:
                continue
            true_idx = class_to_idx[true_label]
            
            # Find image file
            img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, img_id + '.png')
            if not os.path.exists(img_path):
                continue
            
            try:
                raw_img = Image.open(img_path).convert('RGB')
                
                # Apply corruption or pass through clean
                if corruption_type is not None:
                    corrupted = ClinicalCorruptions.apply(raw_img, corruption_type, severity)
                else:
                    corrupted = raw_img
                
                img_tensor = transform(corrupted).unsqueeze(0)
                output = model(img_tensor.to(device))
                pred = output.argmax(dim=1).item()
                
                if pred == true_idx:
                    correct += 1
                total += 1
                
                if idx % 100 == 0 and idx > 0:
                    print(f"    Processed {idx}/{len(test_df)} images...")
            except Exception as e:
                continue
    
    acc = correct / total if total > 0 else 0
    return acc

def main(checkpoint_path):
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Load test split directly (more robust than going through dataloader)
    try:
        splits = pd.read_csv('data/splits/ham10000_hospital_splits.csv')
        test_df = splits[splits['split'] == 'test'].copy()
    except:
        splits = pd.read_csv('data/splits/ham10000_splits.csv')
        test_df = splits[splits['split'] == 'test'].copy()
    
    # Class mapping
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    
    # Image directory
    img_dir = config['data']['ham10000_images']
    
    # Load model
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    print("="*70)
    print("CLINICAL CORRUPTION ROBUSTNESS EVALUATION")
    print("="*70)
    print(f"Test images: {len(test_df)}")
    print("WARNING: This evaluates all test images for each corruption.")
    print("It will take 30-60 minutes on CPU. Go get coffee.")
    print("="*70)
    
    # Baseline
    print("\n[1/13] Evaluating baseline (no corruption)...")
    baseline_acc = evaluate_on_corruption(model, device, test_df, img_dir, None, 0, class_to_idx)
    print(f"Baseline accuracy: {baseline_acc*100:.2f}%")
    
    corruptions = ['hair_occlusion', 'ruler_overlay', 'color_temperature', 'jpeg_artifact']
    severities = [1, 3, 5]
    results = {'baseline': baseline_acc}
    
    count = 2
    for corr in corruptions:
        print(f"\n[{count}/13] {corr.upper()}")
        for sev in severities:
            print(f"  Severity {sev}...")
            acc = evaluate_on_corruption(model, device, test_df, img_dir, corr, sev, class_to_idx)
            drop = baseline_acc - acc
            results[f"{corr}_sev{sev}"] = {'accuracy': acc, 'drop': drop}
            print(f"    Accuracy: {acc*100:.2f}% | Drop: {drop*100:+.1f}%")
            count += 1
    
    # Save
    save_dir = os.path.dirname(checkpoint_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'clinical_corruption_robustness.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY: Accuracy Drop Under Clinical Corruptions")
    print(f"{'='*70}")
    for corr in corruptions:
        for sev in severities:
            key = f"{corr}_sev{sev}"
            if key in results:
                print(f"{corr:20s} sev{sev}: {results[key]['drop']*100:+.1f}% drop")
    print(f"{'='*70}")
    print(f"Saved to {save_dir}/clinical_corruption_robustness.json")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)