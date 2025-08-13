#!/usr/bin/env python3
"""
Training script for underwater image enhancement autoencoder
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.models.unet_autoencoder import UNetAutoencoder, LightweightUNet
from src.models.attention_unet import AttentionUNet, WaterNet
from src.data.dataset import UnderwaterImageDataset, create_data_loaders
from src.utils.losses import UnderwaterEnhancementLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for underwater image enhancement"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() and self.config['hardware']['device'] == 'cuda'
            else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        self.setup_directories()
        self.setup_model()
        self.setup_data()
        self.setup_training()
        self.setup_logging()
        
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.early_stop_patience = 20
    
    def setup_directories(self):
        """Create necessary directories"""
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.log_dir = Path(self.config['logging']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_model(self):
        """Initialize model based on configuration"""
        model_type = self.config['model']['type']
        model_params = {
            'n_channels': self.config['model']['n_channels'],
            'n_classes': self.config['model']['n_classes']
        }
        
        if model_type == 'LightweightUNet':
            model_params['base_features'] = self.config['model']['base_features']
            self.model = LightweightUNet(**model_params)
        elif model_type == 'UNetAutoencoder':
            model_params['base_features'] = self.config['model']['base_features']
            model_params['bilinear'] = self.config['model']['bilinear']
            self.model = UNetAutoencoder(**model_params)
        elif model_type == 'AttentionUNet':
            model_params['base_features'] = self.config['model']['base_features']
            self.model = AttentionUNet(**model_params)
        elif model_type == 'WaterNet':
            model_params['base_features'] = self.config['model']['base_features']
            self.model = WaterNet(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {model_type} with {param_count:,} parameters")
    
    def setup_data(self):
        """Setup data loaders"""
        image_size = tuple(self.config['data']['image_size'])
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])
        
        self.train_loader, self.val_loader = create_data_loaders(
            input_dir=self.config['data']['input_dir'],
            target_dir=self.config['data']['target_dir'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            train_split=self.config['data']['train_split'],
            transform=transform
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Training batches: {len(self.train_loader)}")
        logger.info(f"  Validation batches: {len(self.val_loader)}")
    
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        loss_config = self.config['training']['loss']
        self.criterion = UnderwaterEnhancementLoss(
            lambda_pixel=loss_config['lambda_pixel'],
            lambda_perceptual=loss_config['lambda_perceptual'],
            lambda_color=loss_config['lambda_color'],
            lambda_ssim=loss_config['lambda_ssim']
        ).to(self.device)
        
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if self.config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif self.config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        
        epochs = self.config['training']['epochs']
        if self.config['training']['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif self.config['training']['lr_scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        
        if self.config['hardware']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def setup_logging(self):
        """Setup tensorboard logging"""
        if self.config['logging']['tensorboard']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.writer = SummaryWriter(
                log_dir=self.log_dir / f"run_{timestamp}"
            )
        else:
            self.writer = None
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {'pixel': 0, 'perceptual': 0, 'color': 0, 'ssim': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss, loss_dict = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                loss.backward()
                
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                for key, value in loss_dict.items():
                    if key != 'total':
                        self.writer.add_scalar(f'Train/{key}_loss', value, global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return avg_loss, loss_components
    
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        loss_components = {'pixel': 0, 'perceptual': 0, 'color': 0, 'ssim': 0}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch in pbar:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                loss, loss_dict = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                for key in loss_components:
                    if key in loss_dict:
                        loss_components[key] += loss_dict[key]
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            for key, value in loss_components.items():
                self.writer.add_scalar(f'Val/{key}_loss', value, epoch)
            
            if epoch % 5 == 0:
                self.writer.add_images('Val/Input', inputs[:4], epoch)
                self.writer.add_images('Val/Output', outputs[:4], epoch)
                self.writer.add_images('Val/Target', targets[:4], epoch)
        
        return avg_loss, loss_components
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        
        if epoch % self.config['checkpoint']['save_interval'] == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch
    
    def train(self, resume_path: str = None):
        """Main training loop"""
        start_epoch = 0
        if resume_path:
            start_epoch = self.load_checkpoint(resume_path)
        
        epochs = self.config['training']['epochs']
        logger.info(f"Starting training for {epochs - start_epoch} epochs")
        
        for epoch in range(start_epoch, epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            train_loss, train_components = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f}")
            for key, value in train_components.items():
                logger.info(f"  {key}: {value:.4f}")
            
            if (epoch + 1) % self.config['validation']['interval'] == 0:
                val_loss, val_components = self.validate(epoch)
                logger.info(f"Val Loss: {val_loss:.4f}")
                for key, value in val_components.items():
                    logger.info(f"  {key}: {value:.4f}")
                
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                
                self.save_checkpoint(epoch, val_loss, is_best)
                
                if self.early_stop_counter >= self.early_stop_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            self.scheduler.step()
        
        if self.writer:
            self.writer.close()
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train underwater image enhancement model")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train(args.resume)


if __name__ == "__main__":
    main()