import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml, torch, json, numpy as np
from src.models.baselines import build_model
from src.data.dataloader import get_dataloaders
from src.evaluation.temperature_scaling import TemperatureScaling
from sklearn.metrics import confusion_matrix, classification_report

def main(checkpoint_path):
    with open('configs/experiment_config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    _, val_loader, test_loader, class_names = get_dataloaders(config)
    
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Collect predictions
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, lbls, _ in test_loader:
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
            all_preds.append(probs.argmax(dim=1))
            all_labels.append(lbls)
    
    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    confidences = probs.max(axis=1)
    
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    print("="*70)
    print("CLASS-STRATIFIED TRUST ANALYSIS")
    print("="*70)
    
    # Per-class accuracy and confidence
    for i, cls in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() == 0:
            continue
        cls_acc = (preds[mask] == i).mean()
        cls_conf = confidences[mask].mean()
        cls_rejected = (confidences[mask] < 0.75).mean()  # rejection rate
        
        print(f"\n{cls.upper()} (n={mask.sum()}):")
        print(f"  Accuracy:     {cls_acc*100:.1f}%")
        print(f"  Avg Confidence:{cls_conf:.3f}")
        print(f"  Rejection Rate:{cls_rejected*100:.1f}%")
        
        # CRITICAL: melanoma analysis
        if cls == 'mel':
            mel_correct = (preds[mask] == i).sum()
            mel_total = mask.sum()
            print(f"  *** MELANOMA: {mel_correct}/{mel_total} correct ({cls_acc*100:.1f}%) ***")
            if cls_acc < 0.7:
                print(f"  *** WARNING: Model underperforms on melanoma! ***")
    
    # Save
    save_dir = os.path.dirname(checkpoint_path)
    report = {'class_names': CLASS_NAMES}
    for i, cls in enumerate(CLASS_NAMES):
        mask = labels == i
        report[cls] = {
            'n_samples': int(mask.sum()),
            'accuracy': float((preds[mask] == i).mean()) if mask.sum() > 0 else 0,
            'avg_confidence': float(confidences[mask].mean()) if mask.sum() > 0 else 0
        }
    
    with open(os.path.join(save_dir, 'class_stratified_trust.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSaved to {save_dir}/class_stratified_trust.json")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)