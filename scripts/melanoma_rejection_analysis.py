import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml, torch, numpy as np
from src.models.baselines import build_model
from src.data.dataloader import get_dataloaders

def main(checkpoint_path):
    with open('configs/experiment_config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    _, _, test_loader, _ = get_dataloaders(config)
    
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
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
    mel_idx = 4  # melanoma class index
    
    print("="*70)
    print("MELANOMA-SPECIFIC REJECTION ANALYSIS")
    print("="*70)
    
    # Melanoma cases only
    mel_mask = labels == mel_idx
    mel_conf = confidences[mel_mask]
    mel_pred_correct = (preds[mel_mask] == mel_idx)
    
    print(f"\nTotal melanoma cases: {mel_mask.sum()}")
    print(f"Baseline correct: {mel_pred_correct.sum()}/{mel_mask.sum()} ({mel_pred_correct.mean()*100:.1f}%)")
    
    # What happens if we reject low-confidence melanoma predictions?
    print("\n--- Rejection Gate on Melanoma Cases ---")
    for thresh in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
        accepted = mel_conf >= thresh
        if accepted.sum() == 0:
            continue
        
        acc = mel_pred_correct[accepted].mean()
        cov = accepted.mean()
        rejected_incorrect = (~mel_pred_correct)[~accepted].sum()
        
        print(f"Threshold {thresh:.2f} | Coverage: {cov*100:.1f}% | Accuracy on accepted: {acc*100:.1f}% | Incorrect rejected: {rejected_incorrect}")
    
    # Compare to NV
    print("\n--- NV (benign) for comparison ---")
    nv_mask = labels == 5
    nv_conf = confidences[nv_mask]
    nv_correct = (preds[nv_mask] == 5)
    print(f"Baseline correct: {nv_correct.sum()}/{nv_mask.sum()} ({nv_correct.mean()*100:.1f}%)")
    
    for thresh in [0.75]:
        accepted = nv_conf >= thresh
        acc = nv_correct[accepted].mean()
        print(f"Threshold 0.75 | Coverage: {accepted.mean()*100:.1f}% | Accuracy on accepted: {acc*100:.1f}%")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)