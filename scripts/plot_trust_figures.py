import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("results/trust_validation_results.csv")
mel = df[df["is_melanoma_true"]]

# Fig 1: ECS distribution, correct vs incorrect (all classes)
plt.figure(figsize=(6, 4))
plt.hist(df[df["is_correct"]]["ecs_score"], bins=20, alpha=0.6, label="Correct")
plt.hist(df[~df["is_correct"]]["ecs_score"], bins=20, alpha=0.6, label="Incorrect")
plt.xlabel("ECS Score"); plt.ylabel("Count"); plt.legend()
plt.title("Fig 1: ECS Distribution — All Classes")
plt.savefig("results/fig1_ecs_all.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 2: same, but melanoma only — THE core claim
plt.figure(figsize=(6, 4))
plt.hist(mel[mel["is_correct"]]["ecs_score"], bins=15, alpha=0.6, label="Correct melanoma")
plt.hist(mel[~mel["is_correct"]]["ecs_score"], bins=15, alpha=0.6, label="Incorrect melanoma")
plt.xlabel("ECS Score"); plt.ylabel("Count"); plt.legend()
plt.title("Fig 2: ECS Distribution — Melanoma Only")
plt.savefig("results/fig2_ecs_melanoma.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 3: confidence vs ECS scatter, colored by correctness
plt.figure(figsize=(6, 5))
correct = df[df["is_correct"]]
incorrect = df[~df["is_correct"]]
plt.scatter(correct["confidence"], correct["ecs_score"], alpha=0.4, label="Correct", s=15)
plt.scatter(incorrect["confidence"], incorrect["ecs_score"], alpha=0.4, label="Incorrect", s=15)
plt.xlabel("Model Confidence"); plt.ylabel("ECS Score"); plt.legend()
plt.title("Fig 3: Confidence vs ECS")
plt.savefig("results/fig3_confidence_vs_ecs.png", dpi=150, bbox_inches="tight")
plt.close()

# Fig 4: ROC — confidence-only rejection vs ECS rejection
y_true = (~df["is_correct"]).astype(int)  # 1 = error, what we want to "catch"
fpr_c, tpr_c, _ = roc_curve(y_true, 1 - df["confidence"])
fpr_e, tpr_e, _ = roc_curve(y_true, 1 - df["ecs_score"])
plt.figure(figsize=(6, 5))
plt.plot(fpr_c, tpr_c, label=f"Confidence (AUC={auc(fpr_c, tpr_c):.3f})")
plt.plot(fpr_e, tpr_e, label=f"ECS (AUC={auc(fpr_e, tpr_e):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (catches errors)")
plt.legend(); plt.title("Fig 4: Error-Catching ROC — Confidence vs ECS")
plt.savefig("results/fig4_roc_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved 4 figures to results/. (Fig 5 — Grad-CAM montage — needs a separate visual script, ask me once you're ready for it.)")