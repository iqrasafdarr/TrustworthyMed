import numpy as np

class ClassConditionalTrustScore:
    """
    Trust score that uses different strategies per class.
    NV: confidence alone is sufficient (high baseline accuracy)
    Melanoma: requires ECS + PLEU + confidence (low baseline, confidence unreliable)
    """
    def __init__(self):
        # Class indices
        self.MELANOMA = 4
        self.NV = 5
        
        # Thresholds tuned from data
        self.nv_conf_threshold = 0.75      # works well for benign
        self.mel_ecs_threshold = 0.5       # needs ECS
        self.mel_uncertainty_threshold = 0.3  # MC dropout entropy
    
    def compute(self, predicted_class, confidence, ecs=None, mc_uncertainty=None):
        """
        Returns: trust_score [0-100], recommendation, reason
        """
        if predicted_class == self.NV:
            # Benign mole: confidence is reliable
            if confidence >= self.nv_conf_threshold:
                return {
                    'trust_score': float(confidence * 100),
                    'recommendation': 'ACCEPT',
                    'reason': 'NV with high confidence - reliable'
                }
            else:
                return {
                    'trust_score': float(confidence * 50),
                    'recommendation': 'REVIEW',
                    'reason': 'NV with low confidence - verify'
                }
        
        elif predicted_class == self.MELANOMA:
            # Melanoma: NEVER trust confidence alone
            if ecs is None or mc_uncertainty is None:
                # No ECS available - force review
                return {
                    'trust_score': 25.0,
                    'recommendation': 'FORCE_REVIEW',
                    'reason': 'Melanoma prediction - ECS required for trust assessment'
                }
            
            # Full trust score for melanoma
            ecs_component = ecs if ecs > 0 else 0
            unc_component = 1 - min(mc_uncertainty / 0.5, 1.0)
            
            trust = (confidence * 0.3 + ecs_component * 0.4 + unc_component * 0.3) * 100
            
            if trust > 70 and ecs > self.mel_ecs_threshold and mc_uncertainty < self.mel_uncertainty_threshold:
                return {
                    'trust_score': float(trust),
                    'recommendation': 'ACCEPT_WITH_CAUTION',
                    'reason': 'Melanoma with consistent explanation and low uncertainty'
                }
            else:
                return {
                    'trust_score': float(trust),
                    'recommendation': 'URGENT_DERMATOLOGIST_REVIEW',
                    'reason': 'Melanoma prediction with unstable explanation or high uncertainty - specialist required'
                }
        
        else:
            # Other classes: standard trust score
            trust = confidence * 100
            if trust > 75:
                return {'trust_score': float(trust), 'recommendation': 'ACCEPT', 'reason': 'High confidence'}
            elif trust > 50:
                return {'trust_score': float(trust), 'recommendation': 'REVIEW', 'reason': 'Moderate confidence'}
            else:
                return {'trust_score': float(trust), 'recommendation': 'REJECT', 'reason': 'Low confidence'}