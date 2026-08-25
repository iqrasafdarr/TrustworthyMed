import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample

def bootstrap_metric(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            scores.append(score)
        except:
            continue
    scores = np.array(scores)
    mean = scores.mean()
    std = scores.std()
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return mean, std, (lower, upper)

def bootstrap_classification_report(y_true, y_pred, n_bootstrap=1000):
    results = {}
    results['accuracy'] = bootstrap_metric(y_true, y_pred, accuracy_score, n_bootstrap)
    for avg in ['weighted']:
        def prec(y_t, y_p): 
            return precision_recall_fscore_support(y_t, y_p, average=avg, zero_division=0)[0]
        def rec(y_t, y_p): 
            return precision_recall_fscore_support(y_t, y_p, average=avg, zero_division=0)[1]
        def f1(y_t, y_p): 
            return precision_recall_fscore_support(y_t, y_p, average=avg, zero_division=0)[2]
        results[f'precision_{avg}'] = bootstrap_metric(y_true, y_pred, prec, n_bootstrap)
        results[f'recall_{avg}'] = bootstrap_metric(y_true, y_pred, rec, n_bootstrap)
        results[f'f1_{avg}'] = bootstrap_metric(y_true, y_pred, f1, n_bootstrap)
    return results

def format_ci(result, as_percent=True):
    mean, std, (lo, hi) = result
    if as_percent:
        return f"{mean*100:.2f}% ± {std*100:.2f}% [{lo*100:.2f}, {hi*100:.2f}]"
    return f"{mean:.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]"