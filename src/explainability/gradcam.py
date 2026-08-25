import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """
    Grad-CAM: shows which regions of an image most influenced the model's prediction.
    Works by looking at the gradients flowing into the last convolutional layer.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # These fire automatically during a forward/backward pass
        # and let us "peek" at the target layer's internals.
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image, class_idx=None):
        """
        image: a single image tensor, shape (1, 3, H, W)
        class_idx: which class to explain. If None, uses the model's top prediction.
        """
        self.model.eval()
        output = self.model(image)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop only the score for the class we care about
        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward()

        # Global-average-pool the gradients -> importance weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps, then ReLU (we only care about
        # features that positively support this class)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize the small feature map back up to the original image size
        cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)

        # Normalize to 0-1 so it can be drawn as a heatmap
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx