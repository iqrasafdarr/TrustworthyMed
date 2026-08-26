import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(SkinLesionClassifier, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def load_model(checkpoint_path, device='cpu'):
    model = SkinLesionClassifier(num_classes=7)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_checkpoint.py <path_to_best_model.pth>")
        sys.exit(1)
    
    model = load_model(sys.argv[1])
    # Quick test
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Predicted class: {out.argmax(dim=1).item()}")