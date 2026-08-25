import torch
import numpy as np


class PatchLevelEpistemicUncertainty:
    """
    PLEU: Patch-level uncertainty for rejecting unreliable predictions.
    Splits an image into overlapping patches, runs MC Dropout on each,
    and flags the image for rejection if too many patches are uncertain.
    """

    def __init__(self, model, patch_size=32, stride=16, uncertainty_threshold=0.3, reject_fraction=0.4):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.threshold = uncertainty_threshold
        self.reject_fraction = reject_fraction

    def extract_patches(self, image):
        """Slide a window over the image and cut out overlapping patches."""
        patches = []
        positions = []
        _, H, W = image.shape

        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):
                patch = image[:, y:y + self.patch_size, x:x + self.patch_size]
                patches.append(patch)
                positions.append((y, x))

        return torch.stack(patches), positions

    def compute_uncertainty(self, patches, mc_iterations=10):
        """Run the model on each patch multiple times with dropout ON,
        then measure how much the predictions disagree with each other."""
        self.model.train()  # keep dropout active during inference
        predictions = []

        with torch.no_grad():
            for _ in range(mc_iterations):
                logits = self.model(patches)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)

        predictions = torch.stack(predictions)          # (mc_iterations, N_patches, n_classes)
        mean_probs = predictions.mean(dim=0)             # average prediction per patch
        uncertainty = predictions.std(dim=0).mean(dim=1)  # disagreement per patch

        return uncertainty, mean_probs

    def __call__(self, image):
        patches, positions = self.extract_patches(image)
        uncertainty, probs = self.compute_uncertainty(patches)

        pleu_score = (uncertainty > self.threshold).float().mean().item()
        should_reject = pleu_score > self.reject_fraction

        return {
            "pleu_score": pleu_score,
            "should_reject": should_reject,
            "patch_uncertainties": uncertainty,
            "patch_positions": positions,
        }