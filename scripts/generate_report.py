#!/usr/bin/env python3
import os
import json
from datetime import datetime

def main():
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        print("No results found!")
        return
    
    experiments = []
    for folder in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, folder)
        if not os.path.isdir(path):
            continue
            
        exp = {'name': folder, 'path': path}
        
        test_metrics_path = os.path.join(path, 'test_metrics.json')
        if os.path.exists(test_metrics_path):
            with open(test_metrics_path) as f:
                exp['test'] = json.load(f)
        
        ext_metrics_path = os.path.join(path, 'external_metrics.json')
        if os.path.exists(ext_metrics_path):
            with open(ext_metrics_path) as f:
                exp['external'] = json.load(f)
        
        experiments.append(exp)
    
    report = []
    report.append("# TrustworthyMed Experiment Results\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    report.append("## Summary\n")
    report.append("| Experiment | Test Acc | Test ECE | Ext Acc | Ext ECE |")
    report.append("|------------|----------|----------|---------|---------|")
    
    for exp in experiments:
        test_acc = f"{exp['test']['accuracy']*100:.2f}%" if 'test' in exp else "N/A"
        test_ece = f"{exp['test']['ece']:.4f}" if 'test' in exp else "N/A"
        ext_acc = f"{exp['external']['accuracy']*100:.2f}%" if 'external' in exp else "N/A"
        ext_ece = f"{exp['external']['ece']:.4f}" if 'external' in exp else "N/A"
        report.append(f"| {exp['name']} | {test_acc} | {test_ece} | {ext_acc} | {ext_ece} |")
    
    report.append("\n")
    
    for exp in experiments:
        report.append(f"## {exp['name']}\n")
        report.append(f"Path: `{exp['path']}`\n")
        
        if 'test' in exp:
            report.append("### In-Domain Test (HAM10000)\n")
            for k, v in exp['test'].items():
                report.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
            report.append("")
        
        if 'external' in exp:
            report.append("### External Validation (BCN20000)\n")
            for k, v in exp['external'].items():
                report.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
            report.append("")
        
        report.append("---\n")
    
    report_path = os.path.join(results_dir, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated: {report_path}")
    print('\n'.join(report))

if __name__ == '__main__':
    main()