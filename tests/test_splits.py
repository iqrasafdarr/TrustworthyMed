"""
Smoke test for create_splits.py's grouped_stratified_split().
Uses synthetic data (no real dataset needed) to confirm:
  1. No lesion_id ever appears in more than one split.
  2. Split sizes are roughly proportional to requested ratios.
  3. Multi-image lesions (the actual leakage risk) stay intact.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from create_splits import grouped_stratified_split  # noqa: E402


def make_synthetic_ham10000(n_lesions=2000, seed=42):
    rng = np.random.default_rng(seed)
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    # nv is majority class in real HAM10000 - mimic that imbalance
    class_probs = [0.03, 0.05, 0.11, 0.01, 0.11, 0.67, 0.02]

    rows = []
    for lesion_idx in range(n_lesions):
        lesion_id = f"HAM_{lesion_idx:05d}"
        dx = rng.choice(classes, p=class_probs)
        n_images = rng.choice([1, 1, 1, 2, 2, 3], size=1)[0]  # most lesions have 1 image
        for img_idx in range(n_images):
            rows.append({
                "lesion_id": lesion_id,
                "image_id": f"{lesion_id}_{img_idx}",
                "dx": dx,
            })
    return pd.DataFrame(rows)


def test_no_lesion_leakage():
    df = make_synthetic_ham10000()
    result = grouped_stratified_split(
        df, group_col="lesion_id", stratify_col="dx",
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42,
    )
    leak_check = result.groupby("lesion_id")["split"].nunique()
    assert (leak_check == 1).all(), "Found lesion_id(s) spanning multiple splits"
    print(f"[PASS] No leakage across {result['lesion_id'].nunique()} lesions "
          f"({len(result)} images).")


def test_split_proportions_reasonable():
    df = make_synthetic_ham10000()
    result = grouped_stratified_split(
        df, group_col="lesion_id", stratify_col="dx",
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42,
    )
    props = result["split"].value_counts(normalize=True)
    assert 0.60 < props["train"] < 0.80, f"train proportion off: {props['train']:.2f}"
    assert 0.08 < props["val"] < 0.22, f"val proportion off: {props['val']:.2f}"
    assert 0.08 < props["test"] < 0.22, f"test proportion off: {props['test']:.2f}"
    print(f"[PASS] Split proportions reasonable: {dict(props.round(3))}")


def test_multi_image_lesions_stay_together():
    df = make_synthetic_ham10000()
    result = grouped_stratified_split(
        df, group_col="lesion_id", stratify_col="dx",
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42,
    )
    multi_image_lesions = df.groupby("lesion_id").size()
    multi_image_lesions = multi_image_lesions[multi_image_lesions > 1].index
    assert len(multi_image_lesions) > 0, "Synthetic data has no multi-image lesions to test"

    for lesion_id in multi_image_lesions:
        splits_for_lesion = result[result["lesion_id"] == lesion_id]["split"].unique()
        assert len(splits_for_lesion) == 1, f"{lesion_id} split across multiple partitions"
    print(f"[PASS] All {len(multi_image_lesions)} multi-image lesions stayed intact.")


if __name__ == "__main__":
    test_no_lesion_leakage()
    test_split_proportions_reasonable()
    test_multi_image_lesions_stay_together()
    print("\nAll smoke tests passed.")
