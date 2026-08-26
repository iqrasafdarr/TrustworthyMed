import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io

class ClinicalCorruptions:
    """
    Dermatology-specific corruptions for robustness evaluation.
    These simulate real clinical artifacts, not generic ImageNet noise.
    """
    
    @staticmethod
    def hair_occlusion(image, severity=3):
        """Simulate hair strands over lesion. Severity 1-5."""
        img = np.array(image).copy()
        h, w = img.shape[:2]
        n_hairs = severity * 15
        
        for _ in range(n_hairs):
            # Random hair line
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            angle = np.random.uniform(0, np.pi)
            length = np.random.randint(20, 80)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            # Draw dark line (hair)
            color = np.random.randint(10, 40)
            thickness = np.random.randint(1, severity + 1)
            
            # Bresenham-like line drawing
            steps = max(abs(x2-x1), abs(y2-y1)) + 1
            for t in range(steps):
                x = int(x1 + (x2-x1) * t / steps)
                y = int(y1 + (y2-y1) * t / steps)
                if 0 <= x < w and 0 <= y < h:
                    for dx in range(-thickness//2, thickness//2+1):
                        for dy in range(-thickness//2, thickness//2+1):
                            if 0 <= x+dx < w and 0 <= y+dy < h:
                                img[y+dy, x+dx] = [color, color, color]
        
        return Image.fromarray(img)
    
    @staticmethod
    def ruler_overlay(image, severity=3):
        """Simulate ruler marks at image edge (common in dermoscopy)."""
        img = np.array(image).copy()
        h, w = img.shape[:2]
        
        # Add ruler on bottom edge
        ruler_height = int(h * 0.08 * (severity / 3))
        img[h-ruler_height:h, :] = [200, 200, 200]  # white ruler background
        
        # Add mm tick marks
        tick_spacing = w // 20
        for x in range(0, w, tick_spacing):
            tick_h = ruler_height // 2 if x % (tick_spacing * 5) == 0 else ruler_height // 4
            img[h-tick_h:h, x:x+2] = [50, 50, 50]
        
        return Image.fromarray(img)
    
    @staticmethod
    def color_temperature_shift(image, severity=3):
        """Simulate different hospital lighting (warm/cool shift)."""
        factor = 1.0 + (severity * 0.15)  # 1.15 to 1.75
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def jpeg_artifact(image, severity=3):
        """Simulate PACS compression artifacts."""
        quality = max(5, 95 - severity * 15)
        buf = io.BytesIO()
        image.save(buf, 'JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf)
    
    @staticmethod
    def apply(image, corruption_type, severity=3):
        """Apply corruption by name."""
        if corruption_type == 'hair_occlusion':
            return ClinicalCorruptions.hair_occlusion(image, severity)
        elif corruption_type == 'ruler_overlay':
            return ClinicalCorruptions.ruler_overlay(image, severity)
        elif corruption_type == 'color_temperature':
            return ClinicalCorruptions.color_temperature_shift(image, severity)
        elif corruption_type == 'jpeg_artifact':
            return ClinicalCorruptions.jpeg_artifact(image, severity)
        else:
            return image