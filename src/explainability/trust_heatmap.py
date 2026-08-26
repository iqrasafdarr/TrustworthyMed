import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class TrustHeatmap:
    """
    Fuses PLEU (patch-level uncertainty) and ECS (explanation consistency)
    into a spatial trust overlay on the original image.
    
    Green = trustworthy region
    Red = untrustworthy region
    Yellow = ambiguous
    """
    def __init__(self):
        # Custom colormap: red (untrust) -> yellow -> green (trust)
        self.cmap = LinearSegmentedColormap.from_list(
            'trust', ['red', 'yellow', 'green'], N=256
        )
    
    def generate(self, image, pleu_map, ecs_map=None):
        """
        image: [3, H, W] tensor or numpy array
        pleu_map: [H, W] spatial uncertainty (higher = more uncertain)
        ecs_map: [H, W] spatial explanation consistency (higher = more consistent)
                 If None, uses Grad-CAM variance proxy.
        
        Returns: trust_heatmap [H, W] in [0,1], overlay image
        """
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        h, w = image.shape[:2]
        
        # Normalize PLEU to [0,1]
        pleu_norm = (pleu_map - pleu_map.min()) / (pleu_map.max() - pleu_map.min() + 1e-8)
        
        if ecs_map is not None:
            ecs_norm = (ecs_map - ecs_map.min()) / (ecs_map.max() - ecs_map.min() + 1e-8)
            # Trust = low uncertainty AND high explanation consistency
            trust = (1 - pleu_norm) * 0.5 + ecs_norm * 0.5
        else:
            # Without ECS, trust is just inverse uncertainty
            trust = 1 - pleu_norm
        
        trust = np.clip(trust, 0, 1)
        
        # Create overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Trust heatmap
        im = axes[1].imshow(trust, cmap=self.cmap, vmin=0, vmax=1)
        axes[1].set_title('Trust Heatmap\n(Green=Trust, Red=Reject)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
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