#!/usr/bin/env python3
"""Training script for gaze predictor model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
import wandb
from tqdm import tqdm

try:
    from models.gaze_tracking import GazePredictor
    from data.datasets import get_dataloader
except:
    print("Imports not available - placeholder script")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, gaze_targets) in enumerate(pbar):
        images = images.to(device)
        gaze_targets = gaze_targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        gaze_pred = model(images)
        
        # Compute loss
        loss = criterion(gaze_pred, gaze_targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_samples = 0
    angular_errors = []
    
    with torch.no_grad():
        for images, gaze_targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            gaze_targets = gaze_targets.to(device)
            
            # Forward pass
            gaze_pred = model(images)
            
            # Compute loss
            loss = criterion(gaze_pred, gaze_targets)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            # Compute angular error
            angular_error = compute_angular_error(gaze_pred, gaze_targets)
            angular_errors.extend(angular_error.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    avg_angular_error = sum(angular_errors) / len(angular_errors)
    
    return avg_loss, avg_angular_error


def compute_angular_error(pred, target):
    """Compute angular error in degrees."""
    # Normalize vectors
    pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
    
    # Compute dot product
    dot_product = torch.sum(pred_norm * target_norm, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute angle in degrees
    angle = torch.acos(dot_product) * 180.0 / 3.14159
    
    return angle


def main(args):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project="gaze-prediction", config=vars(args))
    
    # Create model
    model = GazePredictor(
        input_channels=3,
        base_channels=64,
        output_dim=3,
        dropout=args.dropout,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create dataloaders
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        split='train',
    )
    
    val_loader = get_dataloader(
        dataset_name=args.dataset,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        split='val',
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_angular_error = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Angular Error: {val_angular_error:.2f}°")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_angular_error': val_angular_error,
                'learning_rate': optimizer.param_groups[0]['lr'],
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(args.checkpoint_dir) / 'best_model.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_angular_error': val_angular_error,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Step scheduler
        scheduler.step()
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train gaze predictor')
    parser.add_argument('--dataset', type=str, default='openeds', help='Dataset name')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    
    args = parser.parse_args()
    main(args)
