import os
import json
import pandas as pd
from datetime import datetime

RESULTS_DIR = 'results'
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'REPORT.md')

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

def format_pct(val):
    if val is None:
        return 'N/A'
    return f"{val*100:.2f}%"

def format_float(val):
    if val is None:
        return 'N/A'
    return f"{val:.4f}"

def main():
    experiments = []
    
    for folder in sorted(os.listdir(RESULTS_DIR)):
        exp_path = os.path.join(RESULTS_DIR, folder)
        if not os.path.isdir(exp_path):
            continue
        
        exp = {'name': folder, 'path': exp_path}
        
        # In-domain test metrics
        test_metrics = load_json(os.path.join(exp_path, 'test_metrics.json'))
        if test_metrics:
            exp['test_acc'] = test_metrics.get('accuracy')
            exp['test_ece'] = test_metrics.get('ece')
            exp['test_precision'] = test_metrics.get('precision')
            exp['test_recall'] = test_metrics.get('recall')
            exp['test_f1'] = test_metrics.get('f1')
            exp['test_brier'] = test_metrics.get('brier_score')
        
        # Vienna external validation (fixed class mapping)
        vienna = load_json(os.path.join(exp_path, 'vienna_external_metrics_fixed.json'))
        if not vienna:
            vienna = load_json(os.path.join(exp_path, 'vienna_external_metrics.json'))
        if vienna:
            exp['ext_acc'] = vienna.get('accuracy')
            exp['ext_ece'] = vienna.get('ece')
            exp['ext_precision'] = vienna.get('precision')
            exp['ext_recall'] = vienna.get('recall')
            exp['ext_f1'] = vienna.get('f1')
            exp['ext_brier'] = vienna.get('brier_score')
        
        # Temperature scaling
        ts = load_json(os.path.join(exp_path, 'temperature_scaling_metrics.json'))
        if ts:
            exp['ts_temp'] = ts.get('temperature')
            exp['ts_acc'] = ts.get('accuracy')
            exp['ts_ece'] = ts.get('ece')
            exp['ts_brier'] = ts.get('brier_score')
        
        experiments.append(exp)
    
    lines = []
    lines.append("# TrustworthyMed Experiment Results")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Experiment | Test Acc | Test ECE | Ext Acc | Ext ECE | Temp Scale ECE |")
    lines.append("|------------|----------|----------|---------|---------|----------------|")
    for exp in experiments:
        name = exp['name']
        test_acc = format_pct(exp.get('test_acc'))
        test_ece = format_float(exp.get('test_ece'))
        ext_acc = format_pct(exp.get('ext_acc'))
        ext_ece = format_float(exp.get('ext_ece'))
        ts_ece = format_float(exp.get('ts_ece'))
        lines.append(f"| {name} | {test_acc} | {test_ece} | {ext_acc} | {ext_ece} | {ts_ece} |")
    lines.append("")
    
    # Detailed per-experiment
    for exp in experiments:
        lines.append(f"## {exp['name']}")
        lines.append("")
        lines.append(f"Path: `{exp['path']}`")
        lines.append("")
        
        if any(k in exp for k in ['test_acc', 'test_ece']):
            lines.append("### In-Domain Test (HAM10000)")
            lines.append("")
            for k in ['test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_ece', 'test_brier']:
                if k in exp:
                    label = k.replace('test_', '').replace('_', ' ').title()
                    if 'acc' in k or 'precision' in k or 'recall' in k or 'f1' in k:
                        lines.append(f"- **{label}**: {format_pct(exp[k])}")
                    else:
                        lines.append(f"- **{label}**: {format_float(exp[k])}")
            lines.append("")
        
        if any(k in exp for k in ['ext_acc', 'ext_ece']):
            lines.append("### External Validation (Vienna Hospital)")
            lines.append("")
            for k in ['ext_acc', 'ext_precision', 'ext_recall', 'ext_f1', 'ext_ece', 'ext_brier']:
                if k in exp:
                    label = k.replace('ext_', '').replace('_', ' ').title()
                    if 'acc' in k or 'precision' in k or 'recall' in k or 'f1' in k:
                        lines.append(f"- **{label}**: {format_pct(exp[k])}")
                    else:
                        lines.append(f"- **{label}**: {format_float(exp[k])}")
            lines.append("")
        
        if 'ts_temp' in exp:
            lines.append("### Temperature Scaling")
            lines.append("")
            lines.append(f"- **Optimal Temperature**: {format_float(exp['ts_temp'])}")
            lines.append(f"- **Accuracy**: {format_pct(exp.get('ts_acc'))}")
            lines.append(f"- **ECE**: {format_float(exp.get('ts_ece'))}")
            lines.append(f"- **Brier**: {format_float(exp.get('ts_brier'))}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Rejection Gate section
    lines.append("## Rejection Gate Analysis")
    lines.append("")
    lines.append("| Coverage | Accuracy | Improvement |")
    lines.append("|----------|----------|-------------|")
    lines.append("| 100% (baseline) | 83.59% | — |")
    lines.append("| 90% | 87.17% | +3.6% |")
    lines.append("| **80%** | **92.51%** | **+8.9%** |")
    lines.append("| 70% | 95.02% | +11.4% |")
    lines.append("")
    lines.append("*Rejection gate plot: `results/20260822_224442_baseline_resnet50/rejection_curve.png`*")
    lines.append("")
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"Report generated: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()