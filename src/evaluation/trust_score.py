import numpy as np

class TrustScore:
    def __init__(self, w_conf=0.3, w_cal=0.2, w_exp=0.3, w_rob=0.2):
        self.weights = np.array([w_conf, w_cal, w_exp, w_rob])
    
    def compute(self, confidence, ece, ecs=None, corruption_drop=None):
        conf_score = confidence
        cal_score = 1 - min(ece / 0.2, 1.0)
        exp_score = ecs if ecs is not None else 0.5
        rob_score = 1 - min(corruption_drop / 0.3, 1.0) if corruption_drop is not None else 0.5
        scores = np.array([conf_score, cal_score, exp_score, rob_score])
        trust = np.dot(scores, self.weights) * 100
        return {
            'trust_score': float(trust),
            'confidence_component': float(conf_score),
            'calibration_component': float(cal_score),
            'explanation_component': float(exp_score),
            'robustness_component': float(rob_score),
            'recommendation': 'HIGH_TRUST' if trust > 80 else ('MODERATE' if trust > 50 else 'REJECT')
        }