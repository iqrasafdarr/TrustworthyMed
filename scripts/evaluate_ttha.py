import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml, torch, numpy as np, pandas as pd, json
from PIL import Image

from src.models.baselines import build_model
from src.data.dataloader import get_transforms
from src.adaptation.test_time_adaptation import TestTimeHospitalAdaptation
from src.evaluation.calibration import evaluate_model

def main(checkpoint_path):
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Load Vienna test data
    splits = pd.read_csv('data/splits/ham10000_hospital_splits.csv')
    vienna_df = splits[splits['split'] == 'test']
    
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    
    img_dir = config['data']['ham10000_images']
    transform = get_transforms('val')
    
    # Load model
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    print("="*70)
    print("TEST-TIME HOSPITAL ADAPTATION (TTHA)")
    print("="*70)
    
    # Baseline (no adaptation)
    print("\n[1/3] Baseline (no adaptation)...")
    
    # Build simple dataloader for Vienna
    from torch.utils.data import DataLoader, Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, df, img_dir, transform):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_id = row['image_id']
            true_label = row['dx']
            
            img_path = os.path.join(self.img_dir, img_id + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, img_id + '.png')
            
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            label = class_to_idx[true_label] if true_label in class_to_idx else 0
            return image, label, img_id
    
    vienna_ds = SimpleDataset(vienna_df, img_dir, transform)
    vienna_loader = DataLoader(vienna_ds, batch_size=16, shuffle=False, num_workers=0)
    
    baseline_metrics, _, _, _, _ = evaluate_model(model, vienna_loader, device, num_classes=7)
    print(f"Baseline: Acc={baseline_metrics['accuracy']*100:.2f}% | ECE={baseline_metrics['ece']:.4f}")
    
    # TTHA: Adapt on first batch of Vienna
    print("\n[2/3] Adapting to Vienna hospital (no labels used)...")
    
    # Get adaptation batch (first 32 images)
    adapt_images = []
    adapt_labels = []
    for i in range(min(32, len(vienna_ds))):
        img, lbl, _ = vienna_ds[i]
        adapt_images.append(img)
        adapt_labels.append(lbl)
    
    adapt_batch = torch.stack(adapt_images).to(device)
    
    ttha = TestTimeHospitalAdaptation(model)
    optimal_temp = ttha.adapt(adapt_batch)
    
    # Re-evaluate after adaptation
    print("\n[3/3] Re-evaluating after TTHA...")
    adapted_metrics, _, _, _, _ = evaluate_model(model, vienna_loader, device, num_classes=7)
    print(f"After TTHA: Acc={adapted_metrics['accuracy']*100:.2f}% | ECE={adapted_metrics['ece']:.4f}")
    
    # Compare
    acc_gain = (adapted_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    ece_gain = baseline_metrics['ece'] - adapted_metrics['ece']
    
    print(f"\n{'='*70}")
    print("TTHA RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {baseline_metrics['accuracy']*100:.2f}% → {adapted_metrics['accuracy']*100:.2f}% ({acc_gain:+.2f}%)")
    print(f"ECE:      {baseline_metrics['ece']:.4f} → {adapted_metrics['ece']:.4f} ({ece_gain:+.4f})")
    print(f"Temperature: {optimal_temp:.4f}")
    print(f"{'='*70}")
    
    # Save
    save_dir = os.path.dirname(checkpoint_path)
    results = {
        'baseline': baseline_metrics,
        'adapted': adapted_metrics,
        'accuracy_gain': float(acc_gain),
        'ece_gain': float(ece_gain),
        'optimal_temperature': float(optimal_temp)
    }
    
    with open(os.path.join(save_dir, 'ttha_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {save_dir}/ttha_results.json")
    
    # Reset model to original state
    ttha.reset()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    main(args.checkpoint)