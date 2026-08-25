# TrustworthyMed Experiment Results

Generated: 2026-08-25 05:40:37

---

## Summary

| Experiment | Test Acc | Test ECE | Ext Acc | Ext ECE | Temp Scale ECE |
|------------|----------|----------|---------|---------|----------------|
| 20260822_163851_baseline_resnet50 | N/A | N/A | N/A | N/A | N/A |
| 20260822_165115_baseline_resnet50 | N/A | N/A | N/A | N/A | N/A |
| 20260822_165720_baseline_resnet50 | 0.00% | 0.1952 | N/A | N/A | N/A |
| 20260822_200012_baseline_resnet50 | N/A | N/A | N/A | N/A | N/A |
| 20260822_222212_baseline_resnet50 | N/A | N/A | N/A | N/A | N/A |
| 20260822_224442_baseline_resnet50 | 83.59% | 0.0605 | 88.15% | 0.0561 | 0.0515 |
| metrics | N/A | N/A | N/A | N/A | N/A |

## 20260822_163851_baseline_resnet50

Path: `results\20260822_163851_baseline_resnet50`

---

## 20260822_165115_baseline_resnet50

Path: `results\20260822_165115_baseline_resnet50`

---

## 20260822_165720_baseline_resnet50

Path: `results\20260822_165720_baseline_resnet50`

### In-Domain Test (HAM10000)

- **Acc**: 0.00%
- **Precision**: 0.00%
- **Recall**: 0.00%
- **F1**: 0.00%
- **Ece**: 0.1952
- **Brier**: 0.8590

---

## 20260822_200012_baseline_resnet50

Path: `results\20260822_200012_baseline_resnet50`

---

## 20260822_222212_baseline_resnet50

Path: `results\20260822_222212_baseline_resnet50`

---

## 20260822_224442_baseline_resnet50

Path: `results\20260822_224442_baseline_resnet50`

### In-Domain Test (HAM10000)

- **Acc**: 83.59%
- **Precision**: 83.08%
- **Recall**: 83.59%
- **F1**: 82.75%
- **Ece**: 0.0605
- **Brier**: 0.2339

### External Validation (Vienna Hospital)

- **Acc**: 88.15%
- **Precision**: 88.26%
- **Recall**: 88.15%
- **F1**: 86.57%
- **Ece**: 0.0561
- **Brier**: 0.1729

### Temperature Scaling

- **Optimal Temperature**: 1.1311
- **Accuracy**: 83.59%
- **ECE**: 0.0515
- **Brier**: 0.2303

---

## metrics

Path: `results\metrics`

---

## Rejection Gate Analysis

| Coverage | Accuracy | Improvement |
|----------|----------|-------------|
| 100% (baseline) | 83.59% | — |
| 90% | 87.17% | +3.6% |
| **80%** | **92.51%** | **+8.9%** |
| 70% | 95.02% | +11.4% |

*Rejection gate plot: `results/20260822_224442_baseline_resnet50/rejection_curve.png`*
