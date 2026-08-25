import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml, torch, numpy as np
from src.models.baselines import build_model
from src.data.dataloader import get_dataloaders
from src.evaluation.class_conditional_trust import ClassConditionalTrustScore

def main(checkpoint_path):
    with open('configs/experiment_config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    _, _, test_loader, class_names = get_dataloaders(config)
    
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    trust = ClassConditionalTrustScore()
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    print("="*70)
    print("CLASS-CONDITIONAL TRUST SCORE DEMO")
    print("="*70)
    print("\nSimulating Malaika's ECS integration (placeholder values):")
    print("-"*70)
    
    # Test on first 20 samples
    count = 0
    with torch.no_grad():
        for images, labels, ids in test_loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confidences = probs.max(dim=1).values.numpy()
            preds = probs.argmax(dim=1).numpy()
            
            for i in range(len(labels)):
                pred_class = int(preds[i])
                true_class = int(labels[i])
                conf = float(confidences[i])
                
                # Simulate ECS values (Malaika's module will provide real ones)
                # Lower ECS on incorrect predictions = unstable explanation
                is_correct = pred_class == true_class
                simulated_ecs = 0.75 if is_correct else 0.25
                simulated_uncertainty = 0.15 if is_correct else 0.45
                
                result = trust.compute(
                    predicted_class=pred_class,
                    confidence=conf,
                    ecs=simulated_ecs,
                    mc_uncertainty=simulated_uncertainty
                )
                
                marker = "✓" if is_correct else "✗"
                print(f"{marker} {CLASS_NAMES[pred_class]:>5} (true:{CLASS_NAMES[true_class]:>5}) | "
                      f"conf:{conf:.2f} | ecs:{simulated_ecs:.2f} | unc:{simulated_uncertainty:.2f} | "
                      f"TRUST:{result['trust_score']:.0f} | {result['recommendation']}")
                
                count += 1
                if count >= 20:
                    break
            
            if count >= 20:
                break
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("Melanoma predictions get 'URGENT_DERMATOLOGIST_REVIEW' unless")
    print("ECS is high (>0.5) AND uncertainty is low (<0.3).")
    print("NV predictions get 'ACCEPT' with confidence >0.75 alone.")
    print("="*70)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)