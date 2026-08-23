import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration: learns a single temperature parameter
    to fix overconfident predictions.
    
    Usage:
        ts = TemperatureScaling()
        temp = ts.fit(val_logits, val_labels)
        calibrated_logits = ts(test_logits)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """
        Optimize temperature on validation set to minimize NLL.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        return self.temperature.item()


def apply_temperature_scaling(model, val_loader, test_loader, device):
    """
    Full pipeline: fit on val set, apply to test set.
    Returns calibrated probabilities and the learned temperature.
    """
    model.eval()
    
    # Collect validation logits and labels
    val_logits_list = []
    val_labels_list = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            val_logits_list.append(logits.cpu())
            val_labels_list.append(labels)
    
    val_logits = torch.cat(val_logits_list)
    val_labels = torch.cat(val_labels_list)
    
    # Fit temperature
    ts = TemperatureScaling()
    optimal_temp = ts.fit(val_logits, val_labels)
    print(f"Learned temperature: {optimal_temp:.4f}")
    
    # Apply to test set
    test_probs_list = []
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            calibrated_logits = ts(logits.cpu())
            probs = torch.softmax(calibrated_logits, dim=1)
            test_probs_list.append(probs)
    
    test_probs = torch.cat(test_probs_list).numpy()
    
    return test_probs, optimal_temp