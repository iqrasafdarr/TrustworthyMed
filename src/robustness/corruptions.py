import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2
import io

class ImageCorruption:
    """
    Apply real-world corruptions to dermoscopy images.
    Novelty: First corruption-aware calibration benchmark in dermatology AI.
    """
    
    @staticmethod
    def gaussian_blur(image, severity=1):
        """Patient movement / out-of-focus"""
        sigma = [1, 2, 3, 4, 5][severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    @staticmethod
    def jpeg_compression(image, severity=1):
        """Telemedicine compression artifacts"""
        quality = [85, 65, 45, 25, 15][severity - 1]
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    @staticmethod
    def brightness_shift(image, severity=1):
        """Different lighting conditions / devices"""
        factor = [0.8, 0.6, 1.4, 1.6, 1.8][severity - 1]
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def contrast_shift(image, severity=1):
        """Over/under-exposed images"""
        factor = [0.8, 0.6, 1.4, 1.6, 1.8][severity - 1]
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def gaussian_noise(image, severity=1):
        """Sensor noise from cheap cameras"""
        img_array = np.array(image).astype(np.float32) / 255.0
        std = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        noise = np.random.normal(0, std, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))


def evaluate_robustness(model, test_loader, device, corruption_fn, severities=[1, 2, 3, 4, 5]):
    """
    Evaluate model as corruption severity increases.
    Returns accuracy and ECE at each severity level.
    """
    from src.evaluation.calibration import evaluate_model
    
    results = []
    
    for sev in severities:
        print(f"\nTesting severity {sev}/5...")
        
        # Apply corruption to entire test set
        corrupted_loader = apply_corruption_to_loader(test_loader, corruption_fn, sev)
        
        # Evaluate
        metrics, probs, preds, labels, ids = evaluate_model(
            model, corrupted_loader, device, num_classes=7
        )
        
        results.append({
            'severity': sev,
            'accuracy': metrics['accuracy'],
            'ece': metrics['ece']
        })
        
        print(f"  Accuracy: {metrics['accuracy']*100:.1f}% | ECE: {metrics['ece']:.4f}")
    
    return results


def apply_corruption_to_loader(dataloader, corruption_fn, severity):
    """
    Wrap dataloader to apply corruption on-the-fly.
    """
    # This is a simplified version - in practice you'd modify the Dataset
    pass


def plot_robustness_curve(results, save_path):
    """
    Plot accuracy and ECE vs corruption severity.
    Key insight: Does calibration degrade faster than accuracy?
    """
    import matplotlib.pyplot as plt
    
    severities = [r['severity'] for r in results]
    accs = [r['accuracy'] for r in results]
    eces = [r['ece'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy curve
    ax1.plot(severities, accs, 'b-o', markersize=8)
    ax1.set_xlabel('Corruption Severity', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy vs Corruption', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # ECE curve
    ax2.plot(severities, eces, 'r-s', markersize=8)
    ax2.set_xlabel('Corruption Severity', fontsize=12)
    ax2.set_ylabel('ECE (Expected Calibration Error)', fontsize=12)
    ax2.set_title('Calibration vs Corruption', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness curve to {save_path}")