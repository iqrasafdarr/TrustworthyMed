import pandas as pd
from pathlib import Path

split_file = Path("data/splits/ham10000_splits.csv")
audit_file = Path("results/phash_duplicate_audit.csv")

df = pd.read_csv(split_file)
audit = pd.read_csv(audit_file)

parent = {x: x for x in df["image_id"]}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for _, r in audit.iterrows():
    union(r["image_id_1"], r["image_id_2"])

groups = {}
for image_id in df["image_id"]:
    groups.setdefault(find(image_id), []).append(image_id)

changed = 0

for ids in groups.values():
    affected = set(ids) & set(audit["image_id_1"]) | set(ids) & set(audit["image_id_2"])
    if affected:
        old_splits = set(df.loc[df["image_id"].isin(ids), "split"])
        if len(old_splits) > 1 or "train" not in old_splits:
            df.loc[df["image_id"].isin(ids), "split"] = "train"
            changed += len(ids)

df.to_csv(split_file, index=False)

print(f"Moved {changed} images into train.")
print("\nNew split sizes:")
print(df["split"].value_counts())
