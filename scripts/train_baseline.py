#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import argparse
from datetime import datetime
import wandb

from src.models.baselines import build_model
from src.data.dataloader import get_dataloaders
from src.training.trainer import Trainer
from src.evaluation.calibration import evaluate_model, plot_reliability_diagram

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(config['training']['seed'])
    
    wandb_run = None
    if config['logging']['use_wandb']:
        wandb_run = wandb.init(
            project=config['logging']['wandb_project'],
            name=f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(config['logging']['save_dir'], 
                           f"{timestamp}_{config['experiment_name']}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {config['experiment_name']}")
    print(f"MODEL: {config['model']['name']}")
    print(f"SAVE DIR: {save_dir}")
    print(f"{'='*60}\n")
    
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    print(f"\nBuilding {config['model']['name']}...")
    model = build_model(config)
    
    print("\nStarting training...")
    trainer = Trainer(model, config, device, wandb_run)
    history = trainer.fit(train_loader, val_loader, save_dir)
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    metrics, probs, preds, labels, ids = evaluate_model(
        model, test_loader, device, 
        num_classes=config['model']['num_classes']
    )
    
    print(f"\nTest Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Test Precision: {metrics['precision']*100:.2f}%")
    print(f"Test Recall: {metrics['recall']*100:.2f}%")
    print(f"Test F1: {metrics['f1']*100:.2f}%")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    
    import json
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    confidences = probs.max(axis=1)
    plot_reliability_diagram(
        confidences, preds, labels,
        os.path.join(save_dir, 'reliability_diagram.png')
    )
    
    if config['uncertainty']['method'] == 'mc_dropout':
        print("\n" + "="*60)
        print("MC DROPOUT UNCERTAINTY EVALUATION")
        print("="*60)
        
        mc_metrics, mc_probs, mc_preds, mc_labels, mc_ids = evaluate_model(
            model, test_loader, device,
            num_classes=config['model']['num_classes'],
            mc_iterations=config['uncertainty']['mc_iterations']
        )
        
        print(f"MC Dropout ECE: {mc_metrics['ece']:.4f}")
        
        with open(os.path.join(save_dir, 'mc_dropout_metrics.json'), 'w') as f:
            json.dump(mc_metrics, f, indent=2)
    
    import pandas as pd
    results_df = pd.DataFrame({
        'image_id': ids,
        'true_label': labels,
        'pred_label': preds,
        'confidence': confidences
    })
    results_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED TO: {save_dir}")
    print(f"{'='*60}")
    
    if wandb_run:
        wandb_run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    args = parser.parse_args()
    main(args.config)