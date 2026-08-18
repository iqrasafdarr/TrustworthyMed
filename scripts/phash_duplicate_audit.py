"""
phash_duplicate_audit.py

Phase 1, Step 1 (secondary QA): audits for near-duplicate images that slip
past lesion_id-based grouping. Published audits of HAM10000 found a small
number of image pairs that are visually near-identical but carry different
lesion_id values in the official metadata - these can still leak across
splits even after grouped splitting.

This script computes perceptual hashes (pHash) for all images and flags
any cross-split pairs within a small Hamming distance. Any flagged pairs
should be manually reviewed and, if confirmed duplicates, moved into the
same split (train, per convention) before training.

Usage:
    python scripts/phash_duplicate_audit.py \
        --splits data/splits/ham10000_splits.csv \
        --images data/raw/ham10000/images/ \
        --threshold 2
"""
import argparse
from itertools import combinations
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm


def compute_hashes(df: pd.DataFrame, images_dir: Path) -> dict:
    hashes = {}
    for image_id in tqdm(df["image_id"], desc="Hashing images"):
        for ext in (".jpg", ".png"):
            path = images_dir / f"{image_id}{ext}"
            if path.exists():
                try:
                    hashes[image_id] = imagehash.phash(Image.open(path))
                except Exception as e:
                    print(f"[WARN] Could not hash {path}: {e}")
                break
    return hashes


def find_cross_split_duplicates(df: pd.DataFrame, hashes: dict, threshold: int) -> pd.DataFrame:
    id_to_split = dict(zip(df["image_id"], df["split"]))
    items = list(hashes.items())
    flagged = []

    # NOTE: O(n^2) - fine for HAM10000's ~10k images (~50M comparisons is
    # slow but tractable in minutes with imagehash's fast Hamming distance).
    # For larger datasets, replace with an LSH/bucket approach.
    for (id1, h1), (id2, h2) in tqdm(
        combinations(items, 2), desc="Comparing pairs", total=len(items) * (len(items) - 1) // 2
    ):
        if id_to_split.get(id1) == id_to_split.get(id2):
            continue  # same split - not a leakage risk even if duplicate
        dist = h1 - h2
        if dist <= threshold:
            flagged.append({
                "image_id_1": id1, "split_1": id_to_split[id1],
                "image_id_2": id2, "split_2": id_to_split[id2],
                "hamming_distance": dist,
            })

    return pd.DataFrame(flagged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--output", type=str, default="results/phash_duplicate_audit.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.splits)
    images_dir = Path(args.images)

    hashes = compute_hashes(df, images_dir)
    print(f"Hashed {len(hashes)}/{len(df)} images.")

    flagged = find_cross_split_duplicates(df, hashes, args.threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(output_path, index=False)

    if len(flagged) == 0:
        print("[OK] No cross-split near-duplicates found above threshold "
              f"{args.threshold}.")
    else:
        print(f"[ACTION REQUIRED] {len(flagged)} cross-split near-duplicate "
              f"pairs found. Review {output_path} and move one side of each "
              f"pair into the same split (train, by convention) before "
              f"training. Do not proceed to model training until resolved.")


if __name__ == "__main__":
    main()
