import torch
import torch.nn as nn
import torchvision.models as models

class SkinLesionClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=7, pretrained=True, dropout=0.5):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_mc_dropout = False
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits
    
    def enable_mc_dropout(self):
        self.use_mc_dropout = True
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                
    def disable_mc_dropout(self):
        self.use_mc_dropout = False
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()

def build_model(config):
    return SkinLesionClassifier(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )