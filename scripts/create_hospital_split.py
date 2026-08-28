import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('data/raw/ham10000/HAM10000_metadata.csv')
df['lesion_id'] = df['lesion_id'].astype(str)
df['image_id'] = df['image_id'].astype(str)

print("Dataset values:")
print(df['dataset'].value_counts())
print()

# FIXED: Use 'vienna_dias' NOT 'vienna_dg'
train_df = df[df['dataset'] != 'vienna_dias'].copy()
test_df = df[df['dataset'] == 'vienna_dias'].copy()

print(f"Train datasets: {sorted(train_df['dataset'].unique())}")
print(f"Test dataset: {test_df['dataset'].unique()}")
print(f"Test images: {len(test_df)}")

# Split train by lesion_id (prevent leakage)
train_lesions, val_lesions = train_test_split(
    train_df['lesion_id'].unique().tolist(),
    test_size=0.15,
    random_state=42
)

train_df['split'] = train_df['lesion_id'].apply(
    lambda x: 'val' if x in val_lesions else 'train'
)
test_df['split'] = 'test'

splits = pd.concat([train_df, test_df])
out_path = 'data/splits/ham10000_hospital_splits.csv'
splits.to_csv(out_path, index=False)

print(f"\n{'='*60}")
print(f"Train: {(splits['split']=='train').sum()}")
print(f"Val: {(splits['split']=='val').sum()}")
print(f"Test (External - vienna_dias): {(splits['split']=='test').sum()}")
print(f"Saved to: {out_path}")
print(f"{'='*60}")