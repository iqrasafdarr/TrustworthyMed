# TrustworthyMed

Evaluating Reliability, Calibration, and Explainability of Deep Learning Models for Dermoscopic Skin Lesion Classification Under Distribution Shift.

## Status: Phase 1 — Dataset verification & preparation (complete, awaiting real data)

## Research Question
Does a dermoscopic skin-lesion classifier remain accurate, well-calibrated, and reliably self-aware of its own uncertainty when evaluated on data from an independent institution, and under realistic image corruptions — or does it fail silently (high confidence, wrong answer)?

## Datasets
- **HAM10000** — 10,015 dermoscopic images, 7 classes, CC BY-NC 4.0. Train/val/in-domain test.
- **BCN20000** — external validation set. Independently published dataset (Hospital Clínic de Barcelona), filtered to the 7 classes shared with HAM10000.

### A note on the original plan
The initial design proposed "ISIC-2019" as the external validation set. That was scientifically invalid: ISIC-2019 is an aggregate that includes HAM10000 itself as one of its source datasets, so testing on it would silently re-test on training images. This project uses **BCN20000** instead — a genuinely independent dataset with zero image-ID overlap with HAM10000, verified programmatically (`scripts/prepare_external_validation.py`).

## Leakage prevention
HAM10000 lesions can have multiple images per `lesion_id`. Splitting by image (not lesion) is the most common leakage bug reported in prior audits of this dataset. All splits here are grouped by `lesion_id` (`scripts/create_splits.py`, verified by `tests/test_splits.py`), followed by a secondary perceptual-hash duplicate audit (`scripts/phash_duplicate_audit.py`) to catch near-duplicates not captured by lesion_id metadata alone.

## Setup
```bash
pip install -r requirements.txt --break-system-packages
```

Place data at:
```
data/raw/ham10000/HAM10000_metadata.csv
data/raw/ham10000/images/
data/raw/bcn20000/bcn20000_metadata.csv
data/raw/bcn20000/images/
```

## Phase 1 pipeline
```bash
# 1. Verify HAM10000 structure and lesion cardinality
python scripts/verify_ham10000.py \
    --metadata data/raw/ham10000/HAM10000_metadata.csv \
    --images data/raw/ham10000/images/

# 2. Create leakage-safe lesion-level splits
python scripts/create_splits.py \
    --metadata data/raw/ham10000/HAM10000_metadata.csv \
    --output data/splits/ham10000_splits.csv \
    --seed 42

# 3. Secondary near-duplicate audit
python scripts/phash_duplicate_audit.py \
    --splits data/splits/ham10000_splits.csv \
    --images data/raw/ham10000/images/ \
    --threshold 2

# 4. Prepare and verify external validation set
python scripts/prepare_external_validation.py \
    --ham10000-metadata data/raw/ham10000/HAM10000_metadata.csv \
    --bcn20000-metadata data/raw/bcn20000/bcn20000_metadata.csv \
    --output data/splits/bcn20000_external_test.csv

# Run smoke tests any time
python tests/test_splits.py
```

## Project structure
```
TrustworthyMed/
├── README.md
├── requirements.txt
├── configs/                 # experiment_config.yaml - single source of truth
├── src/
│   ├── data/ models/ training/ evaluation/
│   ├── explainability/ uncertainty/ robustness/
├── scripts/                 # verification, splitting, external validation
├── notebooks/
├── results/
├── figures/
├── reports/
└── tests/
```

## Citation
If reusing HAM10000 or BCN20000, cite:
- Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset. *Scientific Data* 5, 180161 (2018).
- Hernández-Pérez, C. et al. BCN20000: Dermoscopic lesions in the wild. *Scientific Data* 11, 641 (2024).
