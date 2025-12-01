"""
Model saving and loading utilities with metadata.

Handles saving best models to dedicated folder with YAML metadata
for full traceability and reproducibility.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import torch.nn as nn


def save_best_model(
    model: nn.Module,
    model_type: str,
    config: Any,
    metrics: Dict[str, float],
    training_time: float,
    dataset_info: Dict[str, int],
    save_dir: str = 'best_models',
    device: str = 'cuda'
) -> tuple:
    """
    Save best model with timestamped name and YAML metadata.
    
    Creates:
    - {model_type}_{date}_best.pth  (model weights)
    - {model_type}_{date}_best.yaml (full metadata)
    
    Args:
        model: Trained model to save
        model_type: Type of model ('cnn', 'resnet', etc.)
        config: ExperimentConfig used for training
        metrics: Dictionary with best metrics (val_loss, val_mae, etc.)
        training_time: Total training time in seconds
        dataset_info: Dict with train_samples, val_samples, test_samples
        save_dir: Directory to save to
        device: Device used for training
        
    Returns:
        Tuple of (model_path, yaml_path)
    """
    # Create directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d")
    base_name = f"{model_type}_{timestamp}_best"
    model_path = save_path / f"{base_name}.pth"
    yaml_path = save_path / f"{base_name}.yaml"
    
    # Handle duplicate names (append number if exists)
    counter = 1
    while model_path.exists():
        base_name = f"{model_type}_{timestamp}_best_{counter}"
        model_path = save_path / f"{base_name}.pth"
        yaml_path = save_path / f"{base_name}.yaml"
        counter += 1
    
    # Save model weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
    }, model_path)
    
    # Format training time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{seconds}s"
    
    # Build metadata
    metadata = {
        'model': {
            'type': model_type,
            'name': model.__class__.__name__,
            'parameters': sum(p.numel() for p in model.parameters()),
            'architecture': {
                'input_channels': getattr(model, 'input_channels', None),
                'base_channels': getattr(model, 'base_channels', None),
            }
        },
        'training': {
            'experiment_name': config.experiment_name,
            'dataset': {
                'total_samples': sum(dataset_info.values()),
                'train_samples': dataset_info.get('train', 0),
                'val_samples': dataset_info.get('val', 0),
                'test_samples': dataset_info.get('test', 0),
                'representation': config.data.representation,
            },
            'hyperparameters': {
                'batch_size': config.training.batch_size,
                'epochs_trained': metrics.get('epochs_trained', 0),
                'learning_rate': config.training.learning_rate,
                'optimizer': config.training.optimizer,
                'loss_function': config.training.loss_function,
            },
            'device': device,
            'training_time': time_str,
            'training_time_seconds': training_time,
        },
        'results': {
            'best_epoch': metrics.get('best_epoch', 0),
            'best_val_loss': metrics.get('best_val_loss', 0),
            'best_val_mae': metrics.get('best_val_mae', 0),
            'final_train_loss': metrics.get('final_train_loss', 0),
            'final_val_loss': metrics.get('final_val_loss', 0),
        },
        'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_path': str(model_path),
    }
    
    # Add model-specific config if available
    if hasattr(config, 'model'):
        if hasattr(config.model, 'dropout_conv'):
            metadata['model']['architecture']['dropout_conv'] = config.model.dropout_conv
        if hasattr(config.model, 'dropout_fc'):
            metadata['model']['architecture']['dropout_fc'] = config.model.dropout_fc
    
    # Save YAML metadata
    with open(yaml_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    return str(model_path), str(yaml_path)


def load_best_model(
    model_path: str,
    model_class: type,
    device: str = 'cpu'
) -> tuple:
    """
    Load a best model and its metadata.
    
    Args:
        model_path: Path to .pth file
        model_class: Model class to instantiate
        device: Device to load model to
        
    Returns:
        Tuple of (model, metadata_dict)
    """
    model_path = Path(model_path)
    yaml_path = model_path.with_suffix('.yaml')
    
    # Load metadata
    metadata = None
    if yaml_path.exists():
        metadata = get_model_info(str(yaml_path))
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with correct parameters
    if metadata and 'model' in metadata:
        arch = metadata['model'].get('architecture', {})
        model = model_class(
            input_channels=arch.get('input_channels', 23),
            base_channels=arch.get('base_channels', 64),
            dropout_conv=arch.get('dropout_conv', 0.3),
            dropout_fc=arch.get('dropout_fc', 0.5)
        )
    else:
        model = model_class()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, metadata


def get_model_info(yaml_path: str) -> Dict[str, Any]:
    """
    Read model metadata from YAML file.
    
    Args:
        yaml_path: Path to YAML metadata file
        
    Returns:
        Dictionary with model metadata
    """
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def list_best_models(save_dir: str = 'best_models') -> list:
    """
    List all saved best models.
    
    Args:
        save_dir: Directory containing best models
        
    Returns:
        List of dicts with model info (name, path, metadata)
    """
    save_path = Path(save_dir)
    if not save_path.exists():
        return []
    
    models = []
    for pth_file in save_path.glob('*.pth'):
        yaml_file = pth_file.with_suffix('.yaml')
        
        model_info = {
            'name': pth_file.stem,
            'model_path': str(pth_file),
            'yaml_path': str(yaml_file) if yaml_file.exists() else None,
        }
        
        if yaml_file.exists():
            metadata = get_model_info(str(yaml_file))
            model_info['type'] = metadata.get('model', {}).get('type', 'unknown')
            model_info['parameters'] = metadata.get('model', {}).get('parameters', 0)
            model_info['best_val_loss'] = metadata.get('results', {}).get('best_val_loss', 0)
            model_info['saved_at'] = metadata.get('saved_at', 'unknown')
        
        models.append(model_info)
    
    # Sort by save date (most recent first)
    models.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
    
    return models


def print_model_info(yaml_path: str):
    """
    Print formatted model information from YAML.
    
    Args:
        yaml_path: Path to YAML metadata file
    """
    from .formatting import print_header, print_metric_table
    
    info = get_model_info(yaml_path)
    
    print_header(f"Model: {info['model']['name']}")
    
    # Model info
    model_stats = {
        'Type': info['model']['type'],
        'Parameters': f"{info['model']['parameters']:,}",
        'Input Channels': info['model']['architecture'].get('input_channels', 'N/A'),
        'Base Channels': info['model']['architecture'].get('base_channels', 'N/A'),
    }
    print_metric_table(model_stats, title="Architecture")
    
    # Training info
    training_stats = {
        'Experiment': info['training']['experiment_name'],
        'Training Time': info['training']['training_time'],
        'Device': info['training']['device'],
        'Batch Size': info['training']['hyperparameters']['batch_size'],
        'Learning Rate': info['training']['hyperparameters']['learning_rate'],
    }
    print_metric_table(training_stats, title="Training")
    
    # Results
    results_stats = {
        'Best Epoch': info['results']['best_epoch'],
        'Best Val Loss': f"{info['results']['best_val_loss']:.6f}",
        'Best Val MAE': f"{info['results']['best_val_mae']:.6f}",
    }
    print_metric_table(results_stats, title="Results")
    
    print(f"\nSaved at: {info['saved_at']}")
    print(f"Model path: {info['model_path']}")

