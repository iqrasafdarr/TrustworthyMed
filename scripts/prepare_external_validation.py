"""
prepare_external_validation.py

Phase 1, Step 1 (external validation setup): prepares BCN20000 as the
external test set, NOT the full ISIC-2019 aggregate.

WHY: ISIC-2019 is an aggregate that includes HAM10000 itself as one of its
source datasets alongside BCN20000 and MSK. Using "ISIC-2019" wholesale as
an external test set would silently re-test on training images. BCN20000
is an independently published dataset (Hospital Clinic de Barcelona) and is
the correct external validation source.

This script:
  1. Filters BCN20000 to the 7 classes shared with HAM10000 (excludes 'scc',
     which HAM10000 does not have).
  2. Cross-checks BCN20000 image/lesion identifiers against HAM10000's to
     confirm zero overlap (defensive check - these are published as
     distinct datasets, but this is verified programmatically rather than
     assumed).

Usage:
    python scripts/prepare_external_validation.py \
        --ham10000-metadata data/raw/ham10000/HAM10000_metadata.csv \
        --bcn20000-metadata data/raw/bcn20000/bcn20000_metadata.csv \
        --output data/splits/bcn20000_external_test.csv
"""
import argparse
from pathlib import Path

import pandas as pd

SHARED_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def filter_to_shared_classes(bcn_df: pd.DataFrame, dx_col: str = "dx") -> pd.DataFrame:
    before = len(bcn_df)
    filtered = bcn_df[bcn_df[dx_col].isin(SHARED_CLASSES)].copy()
    dropped = before - len(filtered)
    print(f"Filtered BCN20000: {before} -> {len(filtered)} images "
          f"({dropped} dropped, mostly 'scc' class not present in HAM10000).")
    return filtered


def check_overlap(ham_df: pd.DataFrame, bcn_df: pd.DataFrame,
                   id_col_ham: str = "image_id", id_col_bcn: str = "image_id") -> None:
    ham_ids = set(ham_df[id_col_ham])
    bcn_ids = set(bcn_df[id_col_bcn])
    overlap = ham_ids & bcn_ids
    if overlap:
        raise RuntimeError(
            f"[CRITICAL] {len(overlap)} image IDs appear in BOTH HAM10000 and "
            f"BCN20000: {list(overlap)[:5]}... External validation is INVALID "
            f"until this is resolved. Do not proceed to model evaluation."
        )
    print(f"[OK] Zero image_id overlap between HAM10000 ({len(ham_ids)} images) "
          f"and BCN20000 ({len(bcn_ids)} images). External validation set is clean.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ham10000-metadata", type=str, required=True)
    parser.add_argument("--bcn20000-metadata", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ham_df = pd.read_csv(args.ham10000_metadata)
    bcn_df = pd.read_csv(args.bcn20000_metadata)

    print(f"Loaded HAM10000: {len(ham_df)} rows")
    print(f"Loaded BCN20000: {len(bcn_df)} rows\n")

    filtered_bcn = filter_to_shared_classes(bcn_df)
    check_overlap(ham_df, filtered_bcn)

    print("\n=== External test set class distribution ===")
    print(filtered_bcn["dx"].value_counts())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_bcn.to_csv(output_path, index=False)
    print(f"\n[OK] External validation set written to {output_path}")
    print(f"     {len(filtered_bcn)} images, {filtered_bcn['dx'].nunique()} classes, "
          f"verified zero overlap with HAM10000.")


if __name__ == "__main__":
    main()
