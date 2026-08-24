
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.models.baselines import build_model
from src.data.dataloader import get_dataloaders
from src.evaluation.bootstrap_ci import bootstrap_classification_report, format_ci

def main():
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    _, _, test_loader, _ = get_dataloaders(config)
    
    # Load model
    model = build_model(config)
    ckpt = torch.load('results/20260822_224442_baseline_resnet50/best_model.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print("Collecting predictions from test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(labels)
    
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    
    print(f"Collected {len(y_true)} predictions.")
    print('Computing 95% Bootstrap CIs (1000 resamples)...')
    print('This takes ~2 minutes...')
    
    results = bootstrap_classification_report(y_true, y_pred, n_bootstrap=1000)
    
    print('\n' + '='*60)
    print('HAM10000 TEST SET — 95% CONFIDENCE INTERVALS')
    print('='*60)
    print(f'Accuracy:  {format_ci(results["accuracy"])}')
    print(f'Precision: {format_ci(results["precision_weighted"])}')
    print(f'Recall:    {format_ci(results["recall_weighted"])}')
    print(f'F1:        {format_ci(results["f1_weighted"])}')
    print('='*60)
    
    # Save to JSON
    save_dir = 'results/20260822_224442_baseline_resnet50'
    import json
    ci_results = {
        'accuracy': {'mean': results['accuracy'][0], 'std': results['accuracy'][1], 
                     'ci95': [results['accuracy'][2][0], results['accuracy'][2][1]]},
        'precision': {'mean': results['precision_weighted'][0], 'std': results['precision_weighted'][1],
                      'ci95': [results['precision_weighted'][2][0], results['precision_weighted'][2][1]]},
        'recall': {'mean': results['recall_weighted'][0], 'std': results['recall_weighted'][1],
                   'ci95': [results['recall_weighted'][2][0], results['recall_weighted'][2][1]]},
        'f1': {'mean': results['f1_weighted'][0], 'std': results['f1_weighted'][1],
               'ci95': [results['f1_weighted'][2][0], results['f1_weighted'][2][1]]}
    }
    with open(os.path.join(save_dir, 'bootstrap_ci.json'), 'w') as f:
        json.dump(ci_results, f, indent=2)
    print(f"\nSaved to {save_dir}/bootstrap_ci.json")

if __name__ == '__main__':
    main()