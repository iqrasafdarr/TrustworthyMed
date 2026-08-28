#!/usr/bin/env python3
"""
Train a deep ensemble of ResNet-50s for TrustworthyMed.

Trains N models with different random seeds. Each model is identical to the
baseline ResNet-50 (ImageNet pretrained, 7-class head, dropout p=0.5 in head).
Checkpoints are saved to results/ensemble_seed{SEED}/best_model.pth

Usage:
    python scripts/train_ensemble.py --num_models 5 --epochs 50 --lr 1e-4
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────── CONFIG ────────────────────────────
IMAGE_DIR = Path("data/raw/ham10000/images")
SPLIT_CSV = Path("data/splits/ham10000_splits.csv")
METADATA_CSV = Path("data/raw/ham10000/HAM10000_metadata.csv")
OUTPUT_ROOT = Path("results/ensemble")

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HAM10000Dataset(Dataset):
    """Minimal dataset compatible with the TrustworthyMed split format."""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = IMAGE_DIR / f"{row['image_id']}.jpg"
        if not img_path.exists():
            # fallback: some datasets use .png or no extension in filename
            alt = IMAGE_DIR / row["image_id"]
            if alt.with_suffix(".jpg").exists():
                img_path = alt.with_suffix(".jpg")
            elif alt.with_suffix(".png").exists():
                img_path = alt.with_suffix(".png")
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        label = CLASS_NAMES.index(row["dx"])

        if self.transform:
            image = self.transform(image)
        return image, label


def build_model():
    """Baseline ResNet-50 with 2-layer classifier head (dropout p=0.5)."""
    model = resnet50(weights="IMAGENET1K_V2")
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )
    return model


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def train_one_model(seed, args):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load splits ──
    splits = pd.read_csv(SPLIT_CSV)
    meta = pd.read_csv(METADATA_CSV)
    df = splits.merge(meta[["image_id", "dx"]], on="image_id", how="left")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    train_ds = HAM10000Dataset(train_df, transform=get_transforms(is_train=True))
    val_ds = HAM10000Dataset(val_df, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ──
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──
    best_val_loss = float("inf")
    patience_counter = 0
    save_dir = OUTPUT_ROOT / f"seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}/{args.epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        val_acc = correct / total

        print(f"[Seed {seed} Epoch {epoch+1}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  -> Saved new best model to {save_dir / 'best_model.pth'}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  -> Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nSeed {seed} training complete. Best val loss: {best_val_loss:.4f}")
    return save_dir / "best_model.pth"


def main():
    parser = argparse.ArgumentParser(description="Train deep ensemble for TrustworthyMed")
    parser.add_argument("--num_models", type=int, default=5, help="Number of ensemble members (default: 5)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per model (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (default: 7)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seeds (default: 42, 43, 44, 45, 46)")
    args = parser.parse_args()

    if args.seeds is None:
        args.seeds = [42 + i for i in range(args.num_models)]

    print("=" * 60)
    print(f"Training deep ensemble: {len(args.seeds)} models")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)

    checkpoint_paths = []
    for seed in args.seeds:
        ckpt = train_one_model(seed, args)
        checkpoint_paths.append(ckpt)

    print("\n" + "=" * 60)
    print("All ensemble members trained.")
    print("Checkpoints:")
    for p in checkpoint_paths:
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()