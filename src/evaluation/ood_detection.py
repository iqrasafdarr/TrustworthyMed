import torch
import numpy as np

class OODDetector:
    """
    Detects out-of-distribution images using energy score.
    Lower energy = in-distribution. Higher energy = OOD.
    """
    def __init__(self, temperature=1.0, percentile=95):
        self.temperature = temperature
        self.percentile = percentile
        self.threshold = None
    
    def energy_score(self, logits):
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
    
    def fit(self, dataloader, model, device):
        energies = []
        model.eval()
        with torch.no_grad():
            for images, _, _ in dataloader:
                logits = model(images.to(device))
                e = self.energy_score(logits)
                energies.extend(e.cpu().numpy())
        
        self.threshold = float(np.percentile(energies, self.percentile))
        print(f"OOD threshold ({self.percentile}th percentile): {self.threshold:.4f}")
        return self.threshold
    
    def predict(self, logits):
        if self.threshold is None:
            raise ValueError("Run fit() first to set threshold.")
        energy = self.energy_score(logits)
        is_ood = energy > self.threshold
        return is_ood.cpu().numpy(), energy.cpu().numpy()
    
    def evaluate(self, id_logits, ood_logits):
        id_energy = self.energy_score(id_logits).cpu().numpy()
        ood_energy = self.energy_score(ood_logits).cpu().numpy()
        
        id_detected = (id_energy > self.threshold).mean()
        ood_detected = (ood_energy > self.threshold).mean()
        
        return {
            'id_false_positive_rate': float(id_detected),
            'ood_true_positive_rate': float(ood_detected),
            'id_mean_energy': float(id_energy.mean()),
            'ood_mean_energy': float(ood_energy.mean()),
        }