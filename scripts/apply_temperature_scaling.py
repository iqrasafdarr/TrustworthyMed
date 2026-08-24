import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.models.baselines import build_model
from src.data.dataloader import get_dataloaders
from src.evaluation.temperature_scaling import TemperatureScaling
from src.evaluation.calibration import expected_calibration_error, brier_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def main(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Load data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
    
    # Load model
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Collecting validation logits for Temperature Scaling...")
    
    # Collect val logits and labels
    val_logits_list = []
    val_labels_list = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            val_logits_list.append(logits.cpu())
            val_labels_list.append(labels)
    
    val_logits = torch.cat(val_logits_list)
    val_labels = torch.cat(val_labels_list)
    
    # Fit temperature
    ts = TemperatureScaling()
    optimal_temp = ts.fit(val_logits, val_labels)
    print(f"\n{'='*60}")
    print(f"OPTIMAL TEMPERATURE: {optimal_temp:.4f}")
    print(f"{'='*60}")
    
    # Evaluate on test set WITH temperature scaling
    print("\nEvaluating with Temperature Scaling...")
    test_probs_list = []
    test_labels_list = []
    test_preds_list = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            scaled_logits = ts(logits.cpu())
            probs = torch.softmax(scaled_logits, dim=1)
            preds = probs.argmax(dim=1)
            
            test_probs_list.append(probs)
            test_labels_list.append(labels)
            test_preds_list.append(preds)
    
    test_probs = torch.cat(test_probs_list).numpy()
    test_preds = torch.cat(test_preds_list).numpy()
    test_labels = torch.cat(test_labels_list).numpy()
    confidences = test_probs.max(axis=1)
    
    # Metrics
    accuracy = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='weighted', zero_division=0)
    ece = expected_calibration_error(confidences, test_preds, test_labels)
    brier = brier_score(test_probs, test_labels, 7)
    
    print(f"\n{'='*60}")
    print("TEST RESULTS WITH TEMPERATURE SCALING")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1: {f1*100:.2f}%")
    print(f"ECE: {ece:.4f}")
    print(f"Brier: {brier:.4f}")
    
    # Save
    save_dir = os.path.dirname(checkpoint_path)
    results = {
        'temperature': float(optimal_temp),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'ece': float(ece),
        'brier_score': float(brier)
    }
    
    import json
    with open(os.path.join(save_dir, 'temperature_scaling_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {save_dir}/temperature_scaling_metrics.json")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)