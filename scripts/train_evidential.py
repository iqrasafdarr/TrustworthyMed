import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.evidential_model import EvidentialSkinLesionClassifier, evidential_loss
from src.data.dataloader import get_dataloaders
from src.evaluation.calibration import expected_calibration_error

def train_epoch(model, train_loader, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        y_onehot = torch.zeros(labels.size(0), num_classes, device=device)
        y_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        loss = evidential_loss(outputs['alpha'], y_onehot)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs['probs'].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, val_loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            probs = outputs['probs']
            preds = probs.argmax(dim=1)
            uncertainty = outputs['uncertainty']
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_uncertainties.append(uncertainty.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_uncertainties = np.concatenate(all_uncertainties)
    
    accuracy = correct / total
    confidences = all_probs.max(axis=1)
    ece = expected_calibration_error(confidences, all_preds, all_labels)
    
    correct_mask = (all_preds == all_labels)
    unc_correct = all_uncertainties[correct_mask].mean() if correct_mask.sum() > 0 else 0
    unc_incorrect = all_uncertainties[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0
    
    return {
        'accuracy': accuracy,
        'ece': ece,
        'uncertainty_correct': float(unc_correct),
        'uncertainty_incorrect': float(unc_incorrect),
        'uncertainty_separation': float(unc_incorrect - unc_correct)
    }

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    num_classes = config['model']['num_classes']
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
    
    model = EvidentialSkinLesionClassifier(num_classes=num_classes)
    model = model.to(device)
    
    try:
        ckpt = torch.load('results/20260822_224442_baseline_resnet50/best_model.pth', map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() 
                          if k in model_dict and 'backbone' in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print("Initialized backbone from existing ResNet-50 checkpoint.")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
        print("Training from scratch...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    best_val_acc = 0.0
    save_dir = 'results/evidential_resnet50'
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = config['training'].get('num_epochs', config['training'].get('epochs', 10))
    
    print("="*70)
    print("EVIDENTIAL DEEP LEARNING TRAINING")
    print("="*70)
    print("This model natively outputs uncertainty — no MC Dropout needed.")
    print(f"Training for {epochs} epochs on CPU (~10 hours total).")
    print("="*70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, num_classes)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Acc: {val_metrics['accuracy']*100:.2f}% | ECE: {val_metrics['ece']:.4f}")
        print(f"Uncertainty — Correct: {val_metrics['uncertainty_correct']:.4f} | "
              f"Incorrect: {val_metrics['uncertainty_incorrect']:.4f} | "
              f"Separation: {val_metrics['uncertainty_separation']:.4f}")
        
        scheduler.step(train_loss)
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model (val acc: {best_val_acc*100:.2f}%)")
    
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    test_metrics = evaluate(model, test_loader, device, num_classes)
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test ECE: {test_metrics['ece']:.4f}")
    print(f"Uncertainty Separation: {test_metrics['uncertainty_separation']:.4f}")
    print("="*70)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    args = parser.parse_args()
    main(args.config)