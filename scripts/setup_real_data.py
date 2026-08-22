import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Setting up real data splits...")

# ========== HAM10000 ==========
df = pd.read_csv('data/raw/ham10000/HAM10000_metadata.csv')
print(f"Loaded HAM10000: {len(df)} images, {df['lesion_id'].nunique()} unique lesions")

lesions = df[['lesion_id', 'dx']].drop_duplicates()
train_lesions, temp_lesions = train_test_split(
    lesions, test_size=0.30, stratify=lesions['dx'], random_state=42
)
val_lesions, test_lesions = train_test_split(
    temp_lesions, test_size=0.50, stratify=temp_lesions['dx'], random_state=42
)

train_ids = set(train_lesions['lesion_id'])
val_ids = set(val_lesions['lesion_id'])
test_ids = set(test_lesions['lesion_id'])

df['split'] = df['lesion_id'].apply(
    lambda x: 'train' if x in train_ids else ('val' if x in val_ids else 'test')
)

df.to_csv('data/splits/ham10000_splits.csv', index=False)
print(f"\nHAM10000 splits saved:")
print(df['split'].value_counts())

# ========== BCN20000 ==========
bcn_df = pd.read_csv('data/raw/bcn20000/bcn20000_metadata.csv')
print(f"\nLoaded BCN20000: {len(bcn_df)} rows")

# Map diagnosis_3 to HAM10000 short codes
diagnosis_map = {
    'Nevus': 'nv',
    'Melanoma, NOS': 'mel',
    'Melanoma metastasis': 'mel',
    'Basal cell carcinoma': 'bcc',
    'Solar lentigo': 'bkl',
    'Seborrheic keratosis': 'bkl',
    'Solar or actinic keratosis': 'akiec',
    'Dermatofibroma': 'df',
    'Scar': None,  # not in HAM10000
    'Squamous cell carcinoma, NOS': None,  # not in HAM10000
}

bcn_df['dx'] = bcn_df['diagnosis_3'].map(diagnosis_map)
bcn_df = bcn_df.dropna(subset=['dx'])

# Also map from diagnosis_2 for vascular lesions
vascular_mask = bcn_df['diagnosis_2'] == 'Benign soft tissue proliferations - Vascular'
bcn_df.loc[vascular_mask, 'dx'] = 'vasc'

bcn_df = bcn_df.rename(columns={'isic_id': 'image_id'})
bcn_df = bcn_df[['image_id', 'dx']].drop_duplicates()

print(f"\nBCN20000 mapped classes:")
print(bcn_df['dx'].value_counts())

bcn_df.to_csv('data/splits/bcn20000_external_test.csv', index=False)
print(f"\nBCN20000 external test saved: {len(bcn_df)} images")

print("\nSetup complete! You can now train.")
