import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes  # FIXED: store num_classes
    
    def forward(self, x):
        evidence = F.relu(self.fc(x))
        alpha = evidence + 1.0
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        uncertainty = self.num_classes / alpha.sum(dim=1)  # FIXED: use self.num_classes
        return probs, alpha, uncertainty, evidence

class EvidentialSkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.evidential_head = EvidentialLayer(in_features, num_classes)
        self.num_classes = num_classes
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        probs, alpha, uncertainty, evidence = self.evidential_head(features)
        return {
            'logits': None,
            'probs': probs,
            'alpha': alpha,
            'uncertainty': uncertainty,
            'evidence': evidence
        }

def evidential_loss(alpha, y, lambda_reg=0.001):
    S = alpha.sum(dim=1, keepdim=True)
    A = torch.digamma(S) - torch.digamma(alpha)
    loss_cls = (y * A).sum(dim=1).mean()
    
    alpha_tilde = y + (1 - y) * alpha
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    
    kl = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(alpha.size(1)), device=alpha.device))
    kl -= (torch.lgamma(alpha_tilde)).sum(dim=1)
    kl += ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1)
    kl = kl.mean()
    
    loss = loss_cls + lambda_reg * kl
    return loss