import torch
import numpy as np
from itertools import combinations
from scipy.stats import pearsonr

from src.explainability.gradcam import GradCAM


class ExplanationConsistencyScore:
    """
    ECS: measures whether the model's Grad-CAM explanation stays consistent
    across repeated stochastic (dropout-on) forward passes.

    High ECS = model looks at the same region every time -> trustworthy.
    Low ECS  = the "reason" for the prediction jumps around randomly -> reject.
    """

    def __init__(self, model, target_layer, T=10, reject_threshold=0.5):
        self.model = model
        self.gradcam = GradCAM(model, target_layer)
        self.T = T
        self.reject_threshold = reject_threshold

    def _get_fixed_class(self, image):
        """Run T dropout passes, average the softmax probs, and lock in
        the argmax as the class we'll explain every single time."""
        self.model.train()  # dropout ON
        probs_sum = None

        with torch.no_grad():
            for _ in range(self.T):
                logits = self.model(image)
                probs = torch.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

        mean_probs = probs_sum / self.T
        return mean_probs.argmax(dim=1).item()

    def _normalize_flatten(self, cam):
        """Min-max normalize a heatmap to [0,1], then flatten to 1D."""
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.flatten()

    def compute(self, image):
        """
        image: single image tensor, shape (1, 3, H, W)
        Returns: dict with ecs_score and should_reject
        """
        fixed_class = self._get_fixed_class(image)

        heatmaps = []
        self.model.train()  # keep dropout ON for each Grad-CAM pass too
        for _ in range(self.T):
            cam, _ = self.gradcam.generate(image, class_idx=fixed_class)
            heatmaps.append(self._normalize_flatten(cam))

        correlations = []
        for h_a, h_b in combinations(heatmaps, 2):
            r, _ = pearsonr(h_a, h_b)
            r = max(r, 0.0)  # clip negative correlation to 0
            correlations.append(r)

        ecs_score = float(np.mean(correlations))
        should_reject = ecs_score < self.reject_threshold

        return {
            "ecs_score": ecs_score,
            "should_reject": should_reject,
            "fixed_class": fixed_class,
        }

    def double_reject(self, image, mc_uncertainty, uncertainty_threshold=0.3):
        """
        Combines ECS with an external MC-uncertainty score (e.g. from PLEU)
        to flag cases that are both uncertain AND explaining themselves badly.
        """
        ecs_result = self.compute(image)
        is_double_reject = ecs_result["should_reject"] and (mc_uncertainty > uncertainty_threshold)

        return {
            **ecs_result,
            "mc_uncertainty": mc_uncertainty,
            "double_reject": is_double_reject,
        }