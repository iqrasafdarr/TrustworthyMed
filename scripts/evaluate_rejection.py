import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader

from src.models.baselines import build_model
from src.data.dataloader import get_dataloaders
from src.evaluation.rejection import evaluate_with_rejection, find_optimal_threshold, plot_rejection_curve

def main(checkpoint_path):
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cpu')
    _, _, test_loader, _ = get_dataloaders(config)

    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Collect predictions
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls, _ in test_loader:
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
            all_labels.append(lbls)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    # Results
    baseline = (preds == labels).mean()

    print("=" * 60)
    print("REJECTION GATE RESULTS")
    print("=" * 60)
    print(f"Baseline accuracy (no rejection): {baseline*100:.2f}%")

    # Test multiple coverage levels
    for target_cov in [0.9, 0.8, 0.7]:
        threshold, acc = find_optimal_threshold(confidences, preds, labels, target_coverage=target_cov)
        print(f"At {target_cov*100:.0f}% coverage (threshold={threshold:.2f}): Accuracy = {acc*100:.2f}%")

    # Save plot
    save_dir = os.path.dirname(checkpoint_path)
    plot_path = os.path.join(save_dir, 'rejection_curve.png')
    plot_rejection_curve(confidences, preds, labels, plot_path)
    print(f"\nSaved rejection curve to {plot_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)