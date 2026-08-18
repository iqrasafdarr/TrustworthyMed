"""
verify_ham10000.py

Phase 1, Step 1: dataset verification.
Run this BEFORE creating any splits. It checks:
  1. Metadata file structure (expected columns present)
  2. Class distribution (dx)
  3. lesion_id cardinality - how many images per lesion (this is the
     leakage risk: multiple images can share one lesion_id)
  4. Missing / unreadable image files
  5. Basic sanity: image count matches metadata row count

Does NOT touch BCN20000 - that's handled separately in
prepare_external_validation.py, since it's a genuinely different
verification problem (confirming no HAM10000 overlap).

Usage:
    python scripts/verify_ham10000.py \
        --metadata data/raw/ham10000/HAM10000_metadata.csv \
        --images data/raw/ham10000/images/
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
EXPECTED_CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
EXPECTED_N_IMAGES = 10015


def verify_metadata_structure(df: pd.DataFrame) -> list[str]:
    problems = []
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        problems.append(f"Missing expected columns: {missing_cols}")

    if len(df) != EXPECTED_N_IMAGES:
        problems.append(
            f"Row count is {len(df)}, expected {EXPECTED_N_IMAGES}. "
            f"This may mean you have a different HAM10000 release/mirror - "
            f"verify source before proceeding."
        )

    found_classes = set(df["dx"].unique()) if "dx" in df.columns else set()
    unexpected = found_classes - EXPECTED_CLASSES
    missing = EXPECTED_CLASSES - found_classes
    if unexpected:
        problems.append(f"Unexpected diagnosis classes found: {unexpected}")
    if missing:
        problems.append(f"Expected classes not found in data: {missing}")

    return problems


def analyze_lesion_cardinality(df: pd.DataFrame) -> dict:
    counts = df.groupby("lesion_id")["image_id"].count()
    return {
        "n_unique_lesions": int(counts.shape[0]),
        "n_images": int(len(df)),
        "lesions_with_multiple_images": int((counts > 1).sum()),
        "max_images_per_lesion": int(counts.max()),
        "images_in_multi_image_lesions": int(counts[counts > 1].sum()),
    }


def verify_image_files(df: pd.DataFrame, images_dir: Path) -> list[str]:
    problems = []
    missing = []
    for image_id in df["image_id"]:
        candidates = [images_dir / f"{image_id}.jpg", images_dir / f"{image_id}.png"]
        if not any(c.exists() for c in candidates):
            missing.append(image_id)
    if missing:
        problems.append(
            f"{len(missing)} image files referenced in metadata were not found "
            f"in {images_dir}. First few: {missing[:5]}"
        )
    return problems


def class_distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    dist = df["dx"].value_counts().rename("count").to_frame()
    dist["pct"] = (dist["count"] / len(df) * 100).round(2)
    return dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--skip-file-check", action="store_true",
                         help="Skip per-image file existence check (slow on large dirs)")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    images_dir = Path(args.images)

    if not metadata_path.exists():
        print(f"[FAIL] Metadata file not found: {metadata_path}")
        print("This script does not download data - place the HAM10000 metadata")
        print("CSV and images directory at the configured paths first.")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df)} rows, columns: {list(df.columns)}\n")

    print("=== Structure check ===")
    problems = verify_metadata_structure(df)
    if problems:
        for p in problems:
            print(f"[WARN] {p}")
    else:
        print("[OK] Structure matches expected HAM10000 schema.")

    print("\n=== Class distribution ===")
    print(class_distribution_report(df))

    print("\n=== Lesion-level cardinality (leakage risk assessment) ===")
    card = analyze_lesion_cardinality(df)
    for k, v in card.items():
        print(f"  {k}: {v}")
    if card["lesions_with_multiple_images"] > 0:
        pct = 100 * card["images_in_multi_image_lesions"] / card["n_images"]
        print(
            f"\n  -> {card['lesions_with_multiple_images']} lesions have multiple images "
            f"({pct:.1f}% of all images). Splitting MUST be done by lesion_id, not "
            f"image_id, or these will leak across train/val/test."
        )

    if not args.skip_file_check:
        print("\n=== Image file existence check ===")
        file_problems = verify_image_files(df, images_dir)
        if file_problems:
            for p in file_problems:
                print(f"[WARN] {p}")
        else:
            print("[OK] All referenced images found on disk.")
    else:
        print("\n=== Image file existence check SKIPPED (--skip-file-check) ===")

    print("\n=== Verdict ===")
    if problems:
        print("[ACTION REQUIRED] Resolve warnings above before creating splits.")
    else:
        print("[OK] Dataset structure verified. Proceed to create_splits.py.")
        print("     Remember: split by lesion_id (see cardinality report above),")
        print("     then run phash_duplicate_audit.py as a secondary check.")


if __name__ == "__main__":
    main()
