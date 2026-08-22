#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader

from src.models.baselines import build_model
from src.data.dataloader import SkinLesionDataset, get_transforms
from src.evaluation.calibration import evaluate_model, plot_reliability_diagram

def main(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    ext_df = pd.read_csv(config['data']['external_test_file'])
    
    ext_dataset = SkinLesionDataset(
        ext_df, 
        config['data']['bcn20000_images'],
        get_transforms('val')
    )
    ext_loader = DataLoader(ext_dataset, batch_size=config['training']['batch_size'], 
                           shuffle=False, num_workers=config['training']['num_workers'])
    
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Evaluating on BCN20000 ({len(ext_dataset)} images)...")
    
    metrics, probs, preds, labels, ids = evaluate_model(
        model, ext_loader, device,
        num_classes=config['model']['num_classes'],
        mc_iterations=config['uncertainty']['mc_iterations'] if config['uncertainty']['method'] == 'mc_dropout' else 1
    )
    
    print(f"\nExternal Validation Results:")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"Brier: {metrics['brier_score']:.4f}")
    
    save_dir = os.path.dirname(checkpoint_path)
    import json
    with open(os.path.join(save_dir, 'external_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    confidences = probs.max(axis=1)
    plot_reliability_diagram(
        confidences, preds, labels,
        os.path.join(save_dir, 'external_reliability_diagram.png')
    )
    
    results_df = pd.DataFrame({
        'image_id': ids,
        'true_label': labels,
        'pred_label': preds,
        'confidence': confidences
    })
    results_df.to_csv(os.path.join(save_dir, 'external_predictions.csv'), index=False)
    
    print(f"Results saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint', required=True, help='Path to best_model.pth')
    args = parser.parse_args()
    main(args.config, args.checkpoint)