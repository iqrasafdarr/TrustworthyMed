import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml, torch, numpy as np, pandas as pd, json
from PIL import Image

from src.models.baselines import build_model
from src.data.dataloader import get_transforms
from src.explainability.ecs import ExplanationConsistencyScore
from src.evaluation.class_conditional_trust import ClassConditionalTrustScore

def get_mc_uncertainty(model, image, T=10):
    """Get MC Dropout uncertainty as entropy of mean probabilities."""
    model.train()
    probs_sum = None
    with torch.no_grad():
        for _ in range(T):
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
    mean_probs = probs_sum / T
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1).item()
    return entropy, mean_probs.argmax(dim=1).item()

def main(checkpoint_path, n_samples=50):
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Load test data
    try:
        splits = pd.read_csv('data/splits/ham10000_hospital_splits.csv')
        test_df = splits[splits['split'] == 'test']
    except:
        splits = pd.read_csv('data/splits/ham10000_splits.csv')
        test_df = splits[splits['split'] == 'test']
    
    # Class mapping
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    
    img_dir = config['data']['ham10000_images']
    transform = get_transforms('val')
    
    # Load model
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    # Init ECS and Trust Score
    ecs = ExplanationConsistencyScore(model, target_layer='layer4', T=10, reject_threshold=0.5)
    trust = ClassConditionalTrustScore()
    
    print("="*70)
    print("ECS + CLASS-CONDITIONAL TRUST EVALUATION")
    print("="*70)
    print(f"Evaluating on {n_samples} test images (this takes ~10 min on CPU)...")
    print("Each image = 10 dropout passes + 10 Grad-CAM generations")
    print("="*70)
    
    results = []
    
    for idx, row in test_df.head(n_samples).iterrows():
        img_id = row['image_id']
        true_label = row['dx']
        true_idx = class_to_idx.get(true_label, -1)
        
        # Load image
        img_path = os.path.join(img_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id + '.png')
        if not os.path.exists(img_path):
            continue
        
        raw_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(raw_img).unsqueeze(0).to(device)
        
        # Baseline prediction
        model.eval()
        with torch.no_grad():
            baseline_logits = model(img_tensor)
            baseline_probs = torch.softmax(baseline_logits, dim=1)
            pred = baseline_probs.argmax(dim=1).item()
            confidence = baseline_probs.max(dim=1).values.item()
        
        is_correct = (pred == true_idx)
        
        # MC Uncertainty
        mc_unc, mc_pred = get_mc_uncertainty(model, img_tensor, T=10)
        
        # ECS
        ecs_result = ecs.compute(img_tensor)
        ecs_score = ecs_result['ecs_score']
        ecs_reject = ecs_result['should_reject']
        
        # Class-conditional trust
        trust_result = trust.compute(
            predicted_class=pred,
            confidence=confidence,
            ecs=ecs_score,
            mc_uncertainty=mc_unc
        )
        
        results.append({
            'image_id': img_id,
            'true_class': CLASS_NAMES[true_idx] if true_idx >= 0 else 'unknown',
            'pred_class': CLASS_NAMES[pred],
            'correct': is_correct,
            'confidence': confidence,
            'ecs_score': ecs_score,
            'mc_uncertainty': mc_unc,
            'ecs_reject': ecs_reject,
            'trust_score': trust_result['trust_score'],
            'recommendation': trust_result['recommendation']
        })
        
        marker = "✓" if is_correct else "✗"
        print(f"{marker} {img_id} | pred:{CLASS_NAMES[pred]:>5} true:{CLASS_NAMES[true_idx]:>5} | "
              f"conf:{confidence:.2f} ecs:{ecs_score:.3f} unc:{mc_unc:.3f} | "
              f"TRUST:{trust_result['trust_score']:.0f} {trust_result['recommendation']}")
    
    # Analysis
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("ECS CATCHES ERRORS THAT CONFIDENCE MISSES")
    print("="*70)
    
    # Confidence-only rejection at 0.75
    conf_rejected = df[df['confidence'] < 0.75]
    conf_accepted = df[df['confidence'] >= 0.75]
    conf_acc = conf_accepted['correct'].mean() if len(conf_accepted) > 0 else 0
    
    # ECS rejection
    ecs_rejected = df[df['ecs_reject'] == True]
    ecs_accepted = df[df['ecs_reject'] == False]
    ecs_acc = ecs_accepted['correct'].mean() if len(ecs_accepted) > 0 else 0
    
    print(f"\nConfidence-only (thresh=0.75):")
    print(f"  Rejected: {len(conf_rejected)}/{len(df)} | Accepted accuracy: {conf_acc*100:.1f}%")
    
    print(f"\nECS-only (thresh=0.5):")
    print(f"  Rejected: {len(ecs_rejected)}/{len(df)} | Accepted accuracy: {ecs_acc*100:.1f}%")
    
    # Melanoma-specific
    mel_df = df[df['true_class'] == 'mel']
    if len(mel_df) > 0:
        print(f"\n--- MELANOMA ONLY (n={len(mel_df)}) ---")
        mel_conf_acc = mel_df[mel_df['confidence'] >= 0.75]['correct'].mean() if len(mel_df[mel_df['confidence'] >= 0.75]) > 0 else 0
        mel_ecs_acc = mel_df[mel_df['ecs_reject'] == False]['correct'].mean() if len(mel_df[mel_df['ecs_reject'] == False]) > 0 else 0
        print(f"Confidence accepted accuracy: {mel_conf_acc*100:.1f}%")
        print(f"ECS accepted accuracy: {mel_ecs_acc*100:.1f}%")
    
    # Save
    save_dir = os.path.dirname(checkpoint_path)
    df.to_csv(os.path.join(save_dir, 'ecs_trust_evaluation.csv'), index=False)
    print(f"\nSaved to {save_dir}/ecs_trust_evaluation.csv")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n_samples', type=int, default=50)
    args = parser.parse_args()
    main(args.checkpoint, args.n_samples)