import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('paper/figures', exist_ok=True)

# Figure 7: Cost-Sensitive Rejection
fig, ax = plt.subplots(figsize=(6, 4))
classes = ['Uniform\n(0.75)', 'Cost-Sensitive\n(Mel: 0.90)']
mel_acc = [42.9, 25.8]
colors = ['#3498db', '#e74c3c']
bars = ax.bar(classes, mel_acc, color=colors, edgecolor='black', width=0.5)
for bar, acc in zip(bars, mel_acc):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Melanoma Accuracy on Accepted Cases (%)')
ax.set_title('(g) Cost-Sensitive Rejection: Higher Threshold = Worse Accuracy', fontweight='bold')
ax.set_ylim(0, 55)
ax.axhline(y=41.2, color='gray', linestyle='--', label='Baseline (41.2%)')
ax.legend()
plt.tight_layout()
fig.savefig('paper/figures/fig7_cost_sensitive.png', dpi=300, bbox_inches='tight')
fig.savefig('paper/figures/fig7_cost_sensitive.pdf', dpi=300, bbox_inches='tight')
print('Saved: fig7_cost_sensitive')

# Figure 8: OOD Detection
fig, ax = plt.subplots(figsize=(5, 4))
categories = ['ID Images\n(Skin)', 'OOD Images\n(Noise)']
detection_rates = [5.0, 100.0]
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(categories, detection_rates, color=colors, edgecolor='black', width=0.4)
for bar, rate in zip(bars, detection_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Flagged as OOD (%)')
ax.set_title('(h) Out-of-Distribution Detection', fontweight='bold')
ax.set_ylim(0, 115)
ax.text(0.5, 50, '5% false positive\n100% true positive', ha='center', fontsize=9)
plt.tight_layout()
fig.savefig('paper/figures/fig8_ood_detection.png', dpi=300, bbox_inches='tight')
fig.savefig('paper/figures/fig8_ood_detection.pdf', dpi=300, bbox_inches='tight')
print('Saved: fig8_ood_detection')
print('Done!')