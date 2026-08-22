import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        self.class_names = sorted(self.df['dx'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, row['image_id'] + '.png')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, row['image_id'])
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[row['dx']]
        return image, label, row['image_id']

def get_transforms(split='train', img_size=224):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(config):
    splits = pd.read_csv(config['data']['splits_file'])
    
    train_df = splits[splits['split'] == 'train']
    val_df = splits[splits['split'] == 'val']
    test_df = splits[splits['split'] == 'test']
    
    train_ds = SkinLesionDataset(train_df, config['data']['ham10000_images'], 
                                  get_transforms('train'))
    val_ds = SkinLesionDataset(val_df, config['data']['ham10000_images'], 
                                get_transforms('val'))
    test_ds = SkinLesionDataset(test_df, config['data']['ham10000_images'], 
                                 get_transforms('val'))
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['training']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=config['training']['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'],
                             shuffle=False, num_workers=config['training']['num_workers'])
    
    return train_loader, val_loader, test_loader, train_ds.class_names