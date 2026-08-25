import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml, torch, numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.models.baselines import build_model
from src.data.dataloader import get_transforms
from src.explainability.gradcam import GradCAM
from src.explainability.trust_heatmap import TrustHeatmap

def main(checkpoint_path, n_images=3):
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
    
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
    
    img_dir = config['data']['ham10000_images']
    transform = get_transforms('val')
    
    # Load model
    model = build_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    # Init GradCAM and TrustHeatmap
    # For SkinLesionClassifier, the ResNet backbone is at model.backbone
    gradcam = GradCAM(model, target_layer=model.backbone.layer4)
    trust_hm = TrustHeatmap()
    
    print("="*70)
    print("TRUST HEATMAP DEMO")
    print("="*70)
    print(f"Generating trust heatmaps for {n_images} images...")
    print("Green = trustworthy, Red = untrustworthy, Yellow = ambiguous")
    print("="*70)
    
    save_dir = os.path.join(os.path.dirname(checkpoint_path), 'trust_heatmaps')
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, row in test_df.head(n_images).iterrows():
        img_id = row['image_id']
        true_label = row['dx']
        
        img_path = os.path.join(img_dir, img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, img_id + '.png')
        if not os.path.exists(img_path):
            continue
        
        raw_img = Image.open(img_path).convert('RGB')
        img_tensor = transform(raw_img).unsqueeze(0).to(device)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs.max(dim=1).values.item()
        
        # Get Grad-CAM
        cam, _ = gradcam.generate(img_tensor, class_idx=pred)
        
        # Create pseudo-PLEU from CAM (inverse = low activation = uncertain)
        pleu_map = 1 - cam
        
        # Generate trust heatmap
        trust_map, fig = trust_hm.generate(
            image=img_tensor.squeeze(0),
            pleu_map=pleu_map,
            ecs_map=None
        )
        
        # Save
        out_path = os.path.join(save_dir, f'trust_heatmap_{img_id}.png')
        trust_hm.save(fig, out_path)
        
        print(f"  {img_id}: pred={CLASS_NAMES[pred]} true={true_label} conf={confidence:.2f}")
        print(f"    Saved: {out_path}")
    
    print(f"\nAll heatmaps saved to: {save_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n_images', type=int, default=3)
    args = parser.parse_args()
    main(args.checkpoint, args.n_images)