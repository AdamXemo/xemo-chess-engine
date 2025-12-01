"""
Training utility functions.

Helper functions for setting seeds, getting learning rates,
and plotting training curves.
"""

import random
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def save_training_curves(
    history: Dict,
    save_path: str,
    metrics: List[str] = ['mse', 'mae']
):
    """
    Plot and save training curves.
    
    Creates a figure with subplots for each metric showing train/val curves.
    
    Args:
        history: Training history dictionary with 'train', 'val', and 'epochs' keys
        save_path: Path to save the figure
        metrics: List of metrics to plot
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    epochs = history['epochs']
    if not epochs:
        print("No training history to plot")
        return
    
    # Filter metrics that exist in history
    available_metrics = [m for m in metrics if m in history['train'] and history['train'][m]]
    
    if not available_metrics:
        print("No metrics available to plot")
        return
    
    # Create subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Plot training curve
        train_values = history['train'][metric]
        ax.plot(epochs, train_values, label='Train', marker='o', linewidth=2)
        
        # Plot validation curve
        val_values = history['val'][metric]
        ax.plot(epochs, val_values, label='Validation', marker='s', linewidth=2)
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} vs Epoch', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Find best validation epoch
        if val_values:
            best_idx = np.argmin(val_values)
            best_epoch = epochs[best_idx]
            best_value = val_values[best_idx]
            
            # Mark best point
            ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_value:.4f}')
            ax.scatter([best_epoch], [best_value], color='red', s=100, zorder=5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        secs = seconds % 60
        return f"{mins:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours:.0f}h {mins:.0f}m"


def save_checkpoint_info(checkpoint_path: str, info: Dict):
    """
    Save checkpoint information to a text file.
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        info: Dictionary with checkpoint information
    """
    info_path = Path(checkpoint_path).with_suffix('.txt')
    
    with open(info_path, 'w') as f:
        f.write("Checkpoint Information\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in info.items():
            f.write(f"{key}: {value}\n")


def load_checkpoint_safely(checkpoint_path: str, model: torch.nn.Module, 
                          optimizer: torch.optim.Optimizer = None,
                          device: str = 'cpu') -> Dict:
    """
    Safely load a checkpoint with error handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def print_training_summary(
    total_epochs: int,
    best_epoch: int,
    best_val_loss: float,
    final_train_loss: float,
    final_val_loss: float,
    total_time: float
):
    """
    Print a summary of training results.
    
    Args:
        total_epochs: Total epochs trained
        best_epoch: Epoch with best validation loss
        best_val_loss: Best validation loss achieved
        final_train_loss: Final training loss
        final_val_loss: Final validation loss
        total_time: Total training time in seconds
    """
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total Epochs:        {total_epochs}")
    print(f"Best Epoch:          {best_epoch}")
    print(f"Best Val Loss:       {best_val_loss:.6f}")
    print(f"Final Train Loss:    {final_train_loss:.6f}")
    print(f"Final Val Loss:      {final_val_loss:.6f}")
    print(f"Total Time:          {format_time(total_time)}")
    print("=" * 60)

