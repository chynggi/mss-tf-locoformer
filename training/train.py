#!/usr/bin/env python3
# Copyright (C) 2024 MSS-TF-Locoformer
# SPDX-License-Identifier: Apache-2.0

"""
Training script for MSS-TF-Locoformer.

Example usage:
    python training/train.py --config configs/musdb18.yaml
"""

import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.mss_tflocoformer import TFLocoformerMSS
from models.mss_loss import MSSLoss
from data.mss_dataset import MUSDBDataset, collate_fn
from utils.common import (
    set_seed, count_parameters, save_checkpoint,
    load_checkpoint, AverageMeter, get_lr
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MSS-TF-Locoformer')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch.
    
    Args:
        model: TF-Locoformer model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
        
    Returns:
        dict: Training metrics
    """
    model.train()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        mixture = batch['mixture'].to(device)
        targets = {
            k: v.to(device) for k, v in batch.items() 
            if k != 'mixture'
        }
        
        # Convert to mono if stereo
        if mixture.ndim == 3 and mixture.shape[1] == 2:
            mixture = mixture.mean(dim=1)
        
        # Forward pass
        predictions = model(mixture, return_time_domain=True)
        
        # Compute loss
        loss_dict = criterion(predictions, targets)
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), mixture.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{get_lr(optimizer):.6f}'
        })
    
    return {
        'train_loss': loss_meter.avg,
    }


def validate(model, dataloader, criterion, device):
    """Validate the model.
    
    Args:
        model: TF-Locoformer model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device
        
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move data to device
            mixture = batch['mixture'].to(device)
            targets = {
                k: v.to(device) for k, v in batch.items()
                if k != 'mixture'
            }
            
            # Convert to mono if stereo
            if mixture.ndim == 3 and mixture.shape[1] == 2:
                mixture = mixture.mean(dim=1)
            
            # Forward pass
            predictions = model(mixture, return_time_domain=True)
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Update metrics
            loss_meter.update(loss.item(), mixture.size(0))
    
    return {
        'val_loss': loss_meter.avg,
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir or config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MUSDBDataset(
        root_dir=config['dataset']['root_dir'],
        subset='train',
        **{k: v for k, v in config['dataset'].items() if k != 'root_dir' and k != 'name'}
    )
    
    val_dataset = MUSDBDataset(
        root_dir=config['dataset']['root_dir'],
        subset='test',
        sample_rate=config['dataset']['sample_rate'],
        segment_length=None,  # Use full tracks for validation
        sources=config['dataset']['sources'],
        augmentation=False,
        random_chunks=False,
    )
    
    print(f"Train dataset: {len(train_dataset)} tracks")
    print(f"Val dataset: {len(val_dataset)} tracks")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn,
    )
    
    # Create model
    print("Creating model...")
    model = TFLocoformerMSS(**config['model'])
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    criterion = MSSLoss(**config['loss'])
    
    # Create optimizer
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            eps=optimizer_config['eps'],
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
    
    # Create scheduler
    scheduler_config = config['training']['scheduler']
    if scheduler_config['type'] == 'reducelronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config['mode'],
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'],
            min_lr=scheduler_config['min_lr'],
        )
    else:
        scheduler = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validate
        if (epoch + 1) % config['training']['val_interval'] == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            
            # Log metrics
            for key, value in train_metrics.items():
                writer.add_scalar(key, value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(key, value, epoch)
            
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step(val_metrics['val_loss'])
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            if (epoch + 1) % config['training']['save_interval'] == 0 or is_best:
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_epoch{epoch+1}.pth"
                )
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['val_loss'],
                    checkpoint_path,
                    best_val_loss=best_val_loss,
                )
                
                if is_best:
                    best_path = os.path.join(checkpoint_dir, "best_model.pth")
                    save_checkpoint(
                        model, optimizer, epoch, val_metrics['val_loss'],
                        best_path,
                        best_val_loss=best_val_loss,
                    )
    
    print("\nTraining complete!")
    writer.close()


if __name__ == '__main__':
    main()
