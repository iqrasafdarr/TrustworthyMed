import json
import os

RESULTS_DIR = 'results/20260822_224442_baseline_resnet50'
OUT_DIR = 'paper/tables'
os.makedirs(OUT_DIR, exist_ok=True)

def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def pct(x):
    return f"{x*100:.2f}"

def flt(x):
    return f"{x:.4f}"

# Table 1: Main Results
test = load_json('test_metrics.json')
vienna = load_json('vienna_external_metrics_fixed.json')
ts = load_json('temperature_scaling_metrics.json')

t1 = r"""\begin{table}[h]
\centering
\caption{Baseline and External Validation Results}
\begin{tabular}{lccc}
\toprule
Metric & HAM10000 Test & Vienna External & After Temp Scaling \\
\midrule
""" + f"""Accuracy (\%) & {pct(test['accuracy'])} & {pct(vienna['accuracy'])} & {pct(ts['accuracy'])} \\\\
Precision (\%) & {pct(test['precision'])} & {pct(vienna['precision'])} & {pct(ts['precision'])} \\\\
Recall (\%) & {pct(test['recall'])} & {pct(vienna['recall'])} & {pct(ts['recall'])} \\\\
F1 (\%) & {pct(test['f1'])} & {pct(vienna['f1'])} & {pct(ts['f1'])} \\\\
ECE & {flt(test['ece'])} & {flt(vienna['ece'])} & {flt(ts['ece'])} \\\\
Brier & {flt(test['brier_score'])} & {flt(vienna['brier_score'])} & {flt(ts['brier_score'])} \\\\
""" + r"""\bottomrule
\end{tabular}
\label{tab:main_results}
\end{table}
"""

with open(os.path.join(OUT_DIR, 'table1_main_results.tex'), 'w') as f:
    f.write(t1)

# Table 2: Rejection Gate
t2 = r"""\begin{table}[h]
\centering
\caption{Rejection Gate: Accuracy vs. Coverage}
\begin{tabular}{ccc}
\toprule
Coverage (\%) & Accuracy (\%) & Improvement \\
\midrule
""" + """100 (baseline) & 83.59 & — \\\\
90 & 87.17 & +3.6 \\\\
80 & 92.51 & +8.9 \\\\
70 & 95.02 & +11.4 \\\\
""" + r"""\bottomrule
\end{tabular}
\label{tab:rejection_gate}
\end{table}
"""

with open(os.path.join(OUT_DIR, 'table2_rejection_gate.tex'), 'w') as f:
    f.write(t2)

# Table 3: Class-Stratified
t3 = r"""\begin{table}[h]
\centering
\caption{Per-Class Accuracy and Confidence on HAM10000 Test Set}
\begin{tabular}{lcc}
\toprule
Class & Accuracy (\%) & Avg. Confidence \\
\midrule
""" + """akiec & 54.3 & 0.812 \\\\
bcc & 73.2 & 0.838 \\\\
bkl & 66.7 & 0.772 \\\\
df & 75.0 & 0.840 \\\\
\\textbf{mel} & \\textbf{41.2} & \\textbf{0.731} \\\\
nv & 95.6 & 0.947 \\\\
vasc & 94.7 & 0.937 \\\\
""" + r"""\bottomrule
\end{tabular}
\label{tab:class_stratified}
\end{table}
"""

with open(os.path.join(OUT_DIR, 'table3_class_stratified.tex'), 'w') as f:
    f.write(t3)

# Table 4: Cost-Sensitive Rejection
t4 = r"""\begin{table}[h]
\centering
\caption{Cost-Sensitive Rejection: Melanoma-Specific Thresholds}
\begin{tabular}{lcc}
\toprule
Threshold Type & Melanoma Accuracy (\%) & Coverage (\%) \\
\midrule
""" + """Uniform (0.75) & 42.9 & 90.4 \\\\
Cost-Sensitive ($\\sim$0.90) & \\textbf{25.8} & 87.5 \\\\
""" + r"""\bottomrule
\end{tabular}
\label{tab:cost_sensitive}
\end{table}
"""

with open(os.path.join(OUT_DIR, 'table4_cost_sensitive.tex'), 'w') as f:
    f.write(t4)

# Table 5: Clinical Corruptions
t5 = r"""\begin{table}[h]
\centering
\caption{Clinical Corruption Robustness (Vienna External)}
\begin{tabular}{lccc}
\toprule
Corruption & Sev 1 & Sev 3 & Sev 5 \\
\midrule
""" + """Hair Occlusion & -1.4\% & -5.9\% & -8.4\% \\\\
Ruler Overlay & -1.1\% & -0.9\% & -1.1\% \\\\
Color Temperature & +0.5\% & +1.4\% & +1.6\% \\\\
JPEG Artifact & +0.0\% & +0.9\% & +1.4\% \\\\
""" + r"""\bottomrule
\end{tabular}
\label{tab:corruptions}
\end{table}
"""

with open(os.path.join(OUT_DIR, 'table5_corruptions.tex'), 'w') as f:
    f.write(t5)

# Table 6: TTHA
t6 = r"""\begin{table}[h]
\centering
\caption{Test-Time Hospital Adaptation (TTHA) on Vienna}
\begin{tabular}{lcc}
\toprule
Metric & Baseline & After TTHA \\
\midrule
""" + f"""Accuracy (\%) & 88.15 & 88.38 \\\\
ECE & 0.0561 & 0.0577 \\\\
Temperature & 1.0000 & 0.9983 \\\\
""" + r"""\bottomrule
\end{tabular}
\label{tab:ttha}
\end{table}
"""

with open(os.path.join(OUT_DIR, 'table6_ttha.tex'), 'w') as f:
    f.write(t6)

print(f"Generated 6 LaTeX tables in {OUT_DIR}/")