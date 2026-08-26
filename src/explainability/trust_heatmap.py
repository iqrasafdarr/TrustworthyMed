import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class TrustHeatmap:
    def __init__(self):
        self.cmap = LinearSegmentedColormap.from_list(
            'trust', ['red', 'yellow', 'green'], N=256
        )
    
    def generate(self, image, pleu_map, ecs_score=None):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        h, w = image.shape[:2]
        pleu_norm = (pleu_map - pleu_map.min()) / (pleu_map.max() - pleu_map.min() + 1e-8)
        pleu_trust = 1 - pleu_norm
        
        if ecs_score is not None:
            trust = pleu_trust * ecs_score
        else:
            trust = pleu_trust
        
        trust = np.clip(trust, 0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        im = axes[1].imshow(trust, cmap=self.cmap, vmin=0, vmax=1)
        axes[1].set_title('Trust Heatmap\n(Green=Trust, Red=Reject)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        axes[2].imshow(image)
        overlay = axes[2].imshow(trust, cmap=self.cmap, vmin=0, vmax=1, alpha=0.5)
        axes[2].set_title('Trust Overlay')
        axes[2].axis('off')
        plt.colorbar(overlay, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        return trust, fig
    
    def save(self, fig, path):
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved trust heatmap to {path}")