import numpy as np
import matplotlib.pyplot as plt

def evaluate_with_rejection(confidences, predictions, labels, threshold):
    """
    Reject low-confidence predictions.
    
    Returns:
        acc_accepted: accuracy on non-rejected cases
        coverage: % of cases accepted (not rejected)
        acc_rejected: accuracy on rejected cases
    """
    accepted = confidences >= threshold
    rejected = ~accepted
    
    if accepted.sum() == 0:
        return 0.0, 0.0, 0.0
    
    acc_accepted = (predictions[accepted] == labels[accepted]).mean()
    acc_rejected = (predictions[rejected] == labels[rejected]).mean() if rejected.sum() > 0 else 0.0
    coverage = accepted.mean()
    
    return float(acc_accepted), float(coverage), float(acc_rejected)

def find_optimal_threshold(confidences, predictions, labels, target_coverage=0.8):
    """
    Find threshold that gives target coverage with best accuracy.
    
    Example: target_coverage=0.8 means "handle 80% of cases, refer 20%"
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold = 0.5
    best_acc = 0.0
    
    for t in thresholds:
        acc, cov, _ = evaluate_with_rejection(confidences, predictions, labels, t)
        if cov >= target_coverage and acc > best_acc:
            best_acc = acc
            best_threshold = t
    
    return best_threshold, best_acc

def plot_rejection_curve(confidences, predictions, labels, save_path):
    """
    Plot accuracy vs coverage at different thresholds.
    This is your KEY FIGURE for the paper.
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    accuracies = []
    coverages = []
    
    for t in thresholds:
        acc, cov, _ = evaluate_with_rejection(confidences, predictions, labels, t)
        accuracies.append(acc)
        coverages.append(cov)
    
    # Baseline accuracy (no rejection)
    baseline_acc = (predictions == labels).mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(coverages, accuracies, 'b-o', markersize=5, label='With Rejection')
    ax.axhline(y=baseline_acc, color='r', linestyle='--', label='No Rejection')
    ax.set_xlabel('Coverage (% of cases accepted)', fontsize=12)
    ax.set_ylabel('Accuracy on Accepted Cases', fontsize=12)
    ax.set_title('Rejection Curve: Accuracy vs Coverage', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotate sweet spot (80% coverage)
    idx_80 = np.argmin(np.abs(np.array(coverages) - 0.8))
    ax.plot(coverages[idx_80], accuracies[idx_80], 'g*', markersize=15, label=f'80% coverage: {accuracies[idx_80]:.2%}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved rejection curve to {save_path}")

def compute_silent_failures(confidences, predictions, labels, confidence_threshold=0.9):
    """
    Silent failure = high confidence (>0.9) + wrong answer.
    These are the dangerous cases that kill patients.
    """
    high_conf = confidences >= confidence_threshold
    wrong = predictions != labels
    
    silent_failures = high_conf & wrong
    silent_fail_rate = silent_failures.mean()
    
    print(f"\nSilent Failure Analysis (confidence >= {confidence_threshold}):")
    print(f"  Total high-confidence predictions: {high_conf.sum()} ({high_conf.mean():.1%})")
    print(f"  Silent failures: {silent_failures.sum()} ({silent_fail_rate:.1%})")
    print(f"  Accuracy among high-confidence: {(predictions[high_conf] == labels[high_conf]).mean():.1%}")
    
    return silent_fail_rate