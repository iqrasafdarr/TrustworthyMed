import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

def expected_calibration_error(confidences, predictions, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def brier_score(probabilities, labels, num_classes):
    one_hot = np.eye(num_classes)[labels]
    return np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

def evaluate_model(model, dataloader, device, num_classes=7, mc_iterations=1):
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    all_ids = []
    
    if mc_iterations > 1:
        model.enable_mc_dropout()
    
    with torch.no_grad():
        for images, labels, ids in dataloader:
            images = images.to(device)
            
            if mc_iterations > 1:
                outputs_list = []
                for _ in range(mc_iterations):
                    outputs = model(images)
                    outputs_list.append(torch.softmax(outputs, dim=1))
                
                probs = torch.stack(outputs_list).mean(dim=0)
            else:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
            
            preds = probs.argmax(dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_ids.extend(list(ids))
    
    if mc_iterations > 1:
        model.disable_mc_dropout()
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    confidences = all_probs.max(axis=1)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    ece = expected_calibration_error(confidences, all_preds, all_labels)
    brier = brier_score(all_probs, all_labels, num_classes)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'ece': float(ece),
        'brier_score': float(brier)
    }
    
    return metrics, all_probs, all_preds, all_labels, all_ids

def plot_reliability_diagram(confidences, predictions, labels, save_path, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    accuracies = []
    counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        count = in_bin.sum()
        counts.append(count)
        
        if count > 0:
            acc = (predictions[in_bin] == labels[in_bin]).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.bar(bin_centers, accuracies, width=0.06, alpha=0.7, label='Accuracy')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    for i, (center, count) in enumerate(zip(bin_centers, counts)):
        ax.text(center, 0.05, str(int(count)), ha='center', fontsize=8)
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved reliability diagram to {save_path}")