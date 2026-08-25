import numpy as np

class CostSensitiveRejection:
    """
    Rejection thresholds weighted by clinical cost of misclassification.
    Melanoma false negative = death (cost = 100)
    Benign false positive = biopsy (cost = 1)
    """
    def __init__(self):
        self.CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.costs = {
            'mel': 100.0,
            'bcc': 10.0,
            'akiec': 10.0,
            'bkl': 2.0,
            'df': 2.0,
            'nv': 1.0,
            'vasc': 1.0,
        }
    
    def get_threshold(self, predicted_class, base_threshold=0.75):
        class_name = self.CLASS_NAMES[predicted_class]
        cost = self.costs[class_name]
        adjusted = base_threshold + (np.log(cost) / 10)
        return min(adjusted, 0.95)
    
    def should_accept(self, predicted_class, confidence, base_threshold=0.75):
        threshold = self.get_threshold(predicted_class, base_threshold)
        return confidence >= threshold, threshold
    
    def evaluate(self, predictions, confidences, labels, base_threshold=0.75):
        """
        Evaluate cost-sensitive rejection vs uniform rejection.
        predictions: [N] predicted class indices
        confidences: [N] max softmax probabilities
        labels: [N] true class indices
        """
        n = len(predictions)
        
        # Uniform rejection (same threshold for all)
        uniform_accepted = confidences >= base_threshold
        uniform_acc = (predictions[uniform_accepted] == labels[uniform_accepted]).mean() if uniform_accepted.sum() > 0 else 0
        uniform_coverage = uniform_accepted.mean()
        
        # Cost-sensitive rejection
        cs_accepted = np.zeros(n, dtype=bool)
        thresholds_used = []
        for i in range(n):
            accept, thresh = self.should_accept(predictions[i], confidences[i], base_threshold)
            cs_accepted[i] = accept
            thresholds_used.append(thresh)
        
        cs_acc = (predictions[cs_accepted] == labels[cs_accepted]).mean() if cs_accepted.sum() > 0 else 0
        cs_coverage = cs_accepted.mean()
        
        # Melanoma-specific metrics
        mel_mask = labels == 4  # melanoma index
        mel_uniform_acc = (predictions[mel_mask & uniform_accepted] == labels[mel_mask & uniform_accepted]).mean() if (mel_mask & uniform_accepted).sum() > 0 else 0
        mel_cs_acc = (predictions[mel_mask & cs_accepted] == labels[mel_mask & cs_accepted]).mean() if (mel_mask & cs_accepted).sum() > 0 else 0
        
        return {
            'uniform': {'accuracy': float(uniform_acc), 'coverage': float(uniform_coverage)},
            'cost_sensitive': {'accuracy': float(cs_acc), 'coverage': float(cs_coverage)},
            'melanoma': {
                'uniform_accuracy': float(mel_uniform_acc),
                'cost_sensitive_accuracy': float(mel_cs_acc),
                'n_cases': int(mel_mask.sum())
            }
        }