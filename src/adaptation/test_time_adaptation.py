import torch
import torch.nn as nn
import numpy as np
from src.evaluation.temperature_scaling import TemperatureScaling

class TestTimeHospitalAdaptation:
    """
    Adapts a pretrained model to a new hospital's data distribution at test time.
    No labels required. Two steps:
    1. Update BatchNorm statistics using target hospital batch
    2. Re-fit temperature using entropy minimization (no labels)
    
    This is the first TTA method for cross-hospital dermatology.
    """
    def __init__(self, model, n_adaptation_steps=1, lr_bn=0.001, lr_temp=0.01):
        self.model = model
        self.n_adaptation_steps = n_adaptation_steps
        self.lr_bn = lr_bn
        self.lr_temp = lr_temp
        
        # Store original BN stats so we can reset
        self.original_bn_stats = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.original_bn_stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone()
                }
    
    def adapt_batch_norm(self, target_batch):
        """
        Update BN running statistics using target hospital batch.
        target_batch: [B, 3, H, W] tensor
        """
        self.model.train()  # BN uses batch stats in train mode
        
        # Run a few forward passes to update running stats
        with torch.no_grad():
            for _ in range(self.n_adaptation_steps):
                _ = self.model(target_batch)
        
        # Freeze BN stats
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # lock the updated stats
    
    def adapt_temperature(self, target_batch, max_iter=50):
        """
        Fit temperature using entropy minimization (no labels needed).
        Lower entropy = more confident = better calibrated for this batch.
        """
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.Adam([temperature], lr=self.lr_temp)
        
        self.model.eval()
        for _ in range(max_iter):
            optimizer.zero_grad()
            
            with torch.no_grad():
                logits = self.model(target_batch)
            
            # Scale by temperature
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            
            # Entropy minimization loss (no labels needed!)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            
            entropy.backward()
            optimizer.step()
        
        return temperature.item()
    
    def adapt(self, target_batch):
        """Full adaptation: BN + Temperature."""
        print("Adapting BatchNorm to target hospital...")
        self.adapt_batch_norm(target_batch)
        
        print("Fitting temperature via entropy minimization...")
        optimal_temp = self.adapt_temperature(target_batch)
        
        print(f"Adaptation complete. Temperature: {optimal_temp:.4f}")
        return optimal_temp
    
    def reset(self):
        """Reset to original pretrained stats."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in self.original_bn_stats:
                module.running_mean = self.original_bn_stats[name]['running_mean']
                module.running_var = self.original_bn_stats[name]['running_var']