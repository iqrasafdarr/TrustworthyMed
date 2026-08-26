import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml, torch, numpy as np, pandas as pd, json
from PIL import Image

from src.models.baselines import build_model
from src.data.dataloader import get_transforms
from src.evaluation.cost_sensitive_rejection import CostSensitiveRejection
from src.evaluation.ood_detection import OODDetector

def main(checkpoint_path):
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    try:
        splits = pd.read_csv('data/splits/ham10000_hospital_splits.csv')
        test_df = splits[splits['split'] == 'test']
    except:
        splits = pd.read_csv('data/splits/ham10000_splits.csv')
        test_df = splits[splits['split'] == 'test']
    
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    
    img_dir = config['data']['ham10000_images']
    transform = get_transforms('val')
    
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("="*70)
    print("COST-SENSITIVE REJECTION + OOD DETECTION")
    print("="*70)
    
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            img_id = row['image_id']
            true_label = row['dx']
            
            img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, img_id + '.png')
            if not os.path.exists(img_path):
                continue
            
            raw_img = Image.open(img_path).convert('RGB')
            img_tensor = transform(raw_img).unsqueeze(0).to(device)
            
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=1).cpu().numpy())
            all_labels.append(class_to_idx.get(true_label, 0))
    
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.array(all_labels)
    confidences = probs.max(axis=1)
    
    # COST-SENSITIVE REJECTION
    print("\n[1/2] Cost-Sensitive Rejection...")
    csr = CostSensitiveRejection()
    csr_results = csr.evaluate(preds, confidences, labels, base_threshold=0.75)
    
    print(f"\nUniform Rejection (thresh=0.75 for ALL):")
    print(f"  Coverage: {csr_results['uniform']['coverage']*100:.1f}%")
    print(f"  Accuracy: {csr_results['uniform']['accuracy']*100:.1f}%")
    
    print(f"\nCost-Sensitive Rejection:")
    print(f"  Coverage: {csr_results['cost_sensitive']['coverage']*100:.1f}%")
    print(f"  Accuracy: {csr_results['cost_sensitive']['accuracy']*100:.1f}%")
    print(f"  Melanoma accuracy: {csr_results['melanoma']['cost_sensitive_accuracy']*100:.1f}%")
    print(f"  (vs {csr_results['melanoma']['uniform_accuracy']*100:.1f}% uniform)")
    
    # OOD DETECTION
    print("\n[2/2] OOD Detection...")
    
    id_logits_list = []
    with torch.no_grad():
        for idx, row in test_df.head(100).iterrows():
            img_id = row['image_id']
            img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, img_id + '.png')
            if not os.path.exists(img_path):
                continue
            
            raw_img = Image.open(img_path).convert('RGB')
            img_tensor = transform(raw_img).unsqueeze(0).to(device)
            logits = model(img_tensor)
            id_logits_list.append(logits)
    
    id_logits = torch.cat(id_logits_list)
    
    ood_images = torch.randn(50, 3, 224, 224).to(device)
    with torch.no_grad():
        ood_logits = model(ood_images)
    
    ood = OODDetector(temperature=1.0, percentile=95)
    id_energies = ood.energy_score(id_logits).cpu().numpy()
    ood.threshold = float(np.percentile(id_energies, 95))
    
    ood_results = ood.evaluate(id_logits, ood_logits)
    
    print(f"\nOOD Detection:")
    print(f"  Threshold: {ood.threshold:.4f}")
    print(f"  ID false positive rate: {ood_results['id_false_positive_rate']*100:.1f}%")
    print(f"  OOD true positive rate: {ood_results['ood_true_positive_rate']*100:.1f}%")
    print(f"  ID mean energy: {ood_results['id_mean_energy']:.2f}")
    print(f"  OOD mean energy: {ood_results['ood_mean_energy']:.2f}")
    
    save_dir = os.path.dirname(checkpoint_path)
    results = {
        'cost_sensitive_rejection': csr_results,
        'ood_detection': ood_results,
        'ood_threshold': ood.threshold
    }
    
    with open(os.path.join(save_dir, 'cost_sensitive_ood.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {save_dir}/cost_sensitive_ood.json")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)