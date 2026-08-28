import torch
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr
from torchvision import transforms
from PIL import Image

from src.explainability.gradcam import GradCAM


class ExplanationConsistencyScore:
    """
    ECS: measures Grad-CAM explanation consistency across 
    Test-Time Augmentations (TTA) of the input image.
    """

    def __init__(self, model, target_layer, T=10, reject_threshold=0.5):
        self.model = model
        self.gradcam = GradCAM(model, target_layer)
        self.T = T
        self.reject_threshold = reject_threshold
        
        # Standard transform: PIL -> Tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _augment_pil(self, pil_img):
        """Apply random augmentation to PIL image."""
        import random
        # Random horizontal flip
        if random.random() > 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        # Random rotation (-10 to 10 degrees)
        angle = random.uniform(-10, 10)
        pil_img = pil_img.rotate(angle)
        return pil_img
    
    def _get_fixed_class(self, pil_img):
        """Run TTA passes, average probs, lock in argmax."""
        self.model.eval()
        probs_sum = None
        
        with torch.no_grad():
            for _ in range(self.T):
                aug_pil = self._augment_pil(pil_img)
                aug_tensor = self.to_tensor(aug_pil).unsqueeze(0).to(next(self.model.parameters()).device)
                
                logits = self.model(aug_tensor)
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
        
        mean_probs = probs_sum / self.T
        return mean_probs.argmax(dim=1).item()
    
    def _normalize_flatten(self, cam):
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.flatten()
    
    def compute(self, pil_img):
        """
        pil_img: PIL Image (RGB)
        Returns: dict with ecs_score, should_reject, fixed_class
        """
        fixed_class = self._get_fixed_class(pil_img)
        
        heatmaps = []
        self.model.eval()
        
        for _ in range(self.T):
            aug_pil = self._augment_pil(pil_img)
            aug_tensor = self.to_tensor(aug_pil).unsqueeze(0).to(next(self.model.parameters()).device)
            
            cam, _ = self.gradcam.generate(aug_tensor, class_idx=fixed_class)
            heatmaps.append(self._normalize_flatten(cam))
        
        correlations = []
        for h_a, h_b in combinations(heatmaps, 2):
            r, _ = pearsonr(h_a, h_b)
            r = max(r, 0.0)
            correlations.append(r)
        
        ecs_score = float(np.mean(correlations))
        should_reject = ecs_score < self.reject_threshold
        
        return {
            "ecs_score": ecs_score,
            "should_reject": should_reject,
            "fixed_class": fixed_class,
        }
    
    def double_reject(self, pil_img, mc_uncertainty, uncertainty_threshold=0.3):
        ecs_result = self.compute(pil_img)
        is_double_reject = ecs_result["should_reject"] and (mc_uncertainty > uncertainty_threshold)
        return {
            **ecs_result,
            "mc_uncertainty": mc_uncertainty,
            "double_reject": is_double_reject,
        }