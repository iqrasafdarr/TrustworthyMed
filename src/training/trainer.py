import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json

class Trainer:
    def __init__(self, model, config, device, wandb_run=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.wandb_run = wandb_run
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels, _ in tqdm(dataloader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return total_loss / len(dataloader), 100. * correct / total
    
    def fit(self, train_loader, val_loader, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            self.scheduler.step(val_loss)
            
            if self.wandb_run:
                self.wandb_run.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': self.config
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
            
        return self.history