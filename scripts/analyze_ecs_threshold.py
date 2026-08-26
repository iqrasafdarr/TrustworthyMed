import pandas as pd
import numpy as np

df = pd.read_csv("results/ecs_melanoma_results.csv")

correct = df[df['is_correct'] == True]['ecs_score']
wrong = df[df['is_correct'] == False]['ecs_score']

print(f"Correct: mean={correct.mean():.4f}, std={correct.std():.4f}, min={correct.min():.4f}")
print(f"Wrong:   mean={wrong.mean():.4f}, std={wrong.std():.4f}, max={wrong.max():.4f}")
print()

# Test every threshold from 0.5 to 0.95
best_f1 = 0
best_thresh = 0
for t in np.arange(0.50, 0.96, 0.01):
    flagged_wrong = wrong[wrong < t]
    flagged_correct = correct[correct < t]
    
    if len(flagged_wrong) + len(flagged_correct) == 0:
        continue
        
    precision = len(flagged_wrong) / (len(flagged_wrong) + len(flagged_correct))
    recall = len(flagged_wrong) / len(wrong)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"BEST THRESHOLD: τ={best_thresh:.2f} | F1={best_f1:.4f}")
print(f"  At τ={best_thresh:.2f}:")
flagged_wrong = wrong[wrong < best_thresh]
flagged_correct = correct[correct < best_thresh]
print(f"  - Catches {len(flagged_wrong)}/{len(wrong)} misclassified ({100*len(flagged_wrong)/len(wrong):.1f}%)")
print(f"  - False alarms: {len(flagged_correct)}/{len(correct)} correct ({100*len(flagged_correct)/len(correct):.1f}%)")

# Bottom 10% ECS analysis
bottom_10_thresh = df['ecs_score'].quantile(0.10)
bottom_10 = df[df['ecs_score'] <= bottom_10_thresh]
wrong_in_bottom = bottom_10[bottom_10['is_correct'] == False]
print(f"\nBottom 10% ECS (≤{bottom_10_thresh:.4f}): {len(wrong_in_bottom)}/{len(bottom_10)} are misclassified ({100*len(wrong_in_bottom)/len(bottom_10):.1f}%)")