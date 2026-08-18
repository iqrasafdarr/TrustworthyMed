"""
create_splits.py

Phase 1, Step 1 (continued): creates leakage-safe train/val/test splits.

Splits by lesion_id (grouped) while stratifying by dx (diagnosis) as closely
as group-constraints allow. Plain StratifiedShuffleSplit is NOT used here
because it operates at the image level and would split lesions with multiple
images across partitions - exactly the leakage pattern documented for
HAM10000 in prior audits.

Usage:
    python scripts/create_splits.py \
        --metadata data/raw/ham10000/HAM10000_metadata.csv \
        --output data/splits/ham10000_splits.csv \
        --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_stratified_split(df: pd.DataFrame, group_col: str, stratify_col: str,
                              train_ratio: float, val_ratio: float, test_ratio: float,
                              seed: int) -> pd.DataFrame:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df = df.copy()

    # Step 1: split off test set by lesion group
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=df[group_col]))
    trainval_df = df.iloc[trainval_idx]
    test_df = df.iloc[test_idx]

    # Step 2: split remaining into train/val by lesion group
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=seed)
    train_idx, val_idx = next(gss_val.split(trainval_df, groups=trainval_df[group_col]))
    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    result = pd.concat([train_df, val_df, test_df]).sort_index()

    # Hard assertion: no lesion_id appears in more than one split
    leak_check = result.groupby(group_col)["split"].nunique()
    leaking_lesions = leak_check[leak_check > 1]
    if len(leaking_lesions) > 0:
        raise RuntimeError(
            f"Leakage detected: {len(leaking_lesions)} lesion_ids appear in "
            f"multiple splits. This should be impossible with GroupShuffleSplit "
            f"- stop and debug before proceeding."
        )

    return result


def report_split_quality(df: pd.DataFrame, stratify_col: str):
    print("\n=== Split sizes ===")
    print(df["split"].value_counts())

    print(f"\n=== Class balance by split ({stratify_col}) ===")
    ct = pd.crosstab(df[stratify_col], df["split"], normalize="columns") * 100
    print(ct.round(2))

    print("\n=== Unique lesions per split ===")
    print(df.groupby("split")["lesion_id"].nunique())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--group-col", type=str, default="lesion_id")
    parser.add_argument("--stratify-col", type=str, default="dx")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    np.random.seed(args.seed)

    result = grouped_stratified_split(
        df,
        group_col=args.group_col,
        stratify_col=args.stratify_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    report_split_quality(result, args.stratify_col)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\n[OK] Splits written to {output_path}")
    print("Next: run scripts/phash_duplicate_audit.py on this split file")
    print("      to catch near-duplicates that lesion_id metadata misses.")


if __name__ == "__main__":
    main()
