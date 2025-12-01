"""
Metrics tracking and computation for training.

Provides utilities to compute and track metrics like MSE, MAE, RMSE
throughout the training process.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import torch
import numpy as np


def compute_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        MSE value
    """
    return torch.nn.functional.mse_loss(predictions, targets).item()


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        MAE value
    """
    return torch.nn.functional.l1_loss(predictions, targets).item()


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        RMSE value
    """
    mse = torch.nn.functional.mse_loss(predictions, targets).item()
    return np.sqrt(mse)


class MetricsTracker:
    """
    Track and compute metrics over epochs.
    
    Maintains running averages of metrics during training and validation,
    and stores historical values for analysis.
    """
    
    def __init__(self, metrics_list: Optional[List[str]] = None):
        """
        Initialize metrics tracker.
        
        Args:
            metrics_list: List of metric names to track (default: ['mse', 'mae', 'rmse'])
        """
        if metrics_list is None:
            metrics_list = ['mse', 'mae', 'rmse']
        
        self.metrics_list = metrics_list
        self.metric_functions = {
            'mse': compute_mse,
            'mae': compute_mae,
            'rmse': compute_rmse,
        }
        
        # Current epoch accumulators
        self.reset()
        
        # Historical values (epoch-level)
        self.history = {
            'train': {metric: [] for metric in metrics_list},
            'val': {metric: [] for metric in metrics_list},
            'epochs': []
        }
    
    def reset(self):
        """Reset accumulators for new epoch."""
        self.predictions_list = []
        self.targets_list = []
        self.num_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Add batch predictions and targets.
        
        Args:
            predictions: Model predictions for batch
            targets: Ground truth targets for batch
        """
        # Detach and move to CPU
        pred = predictions.detach().cpu()
        targ = targets.detach().cpu()
        
        self.predictions_list.append(pred)
        self.targets_list.append(targ)
        self.num_samples += pred.size(0)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics for current epoch.
        
        Returns:
            Dictionary of metric names to values
        """
        if not self.predictions_list:
            return {metric: 0.0 for metric in self.metrics_list}
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(self.predictions_list, dim=0)
        all_targets = torch.cat(self.targets_list, dim=0)
        
        # Compute each metric
        results = {}
        for metric_name in self.metrics_list:
            if metric_name in self.metric_functions:
                metric_func = self.metric_functions[metric_name]
                results[metric_name] = metric_func(all_predictions, all_targets)
            else:
                results[metric_name] = 0.0
        
        return results
    
    def add_to_history(self, metrics: Dict[str, float], epoch: int, phase: str = 'train'):
        """
        Add computed metrics to history.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch number
            phase: 'train' or 'val'
        """
        if phase not in ['train', 'val']:
            raise ValueError(f"phase must be 'train' or 'val', got '{phase}'")
        
        for metric_name, value in metrics.items():
            if metric_name in self.history[phase]:
                self.history[phase][metric_name].append(value)
        
        # Track epoch numbers (only once per epoch)
        if phase == 'train' and (not self.history['epochs'] or self.history['epochs'][-1] != epoch):
            self.history['epochs'].append(epoch)
    
    def get_history(self) -> Dict:
        """
        Get full training history.
        
        Returns:
            Dictionary containing all historical metrics
        """
        return self.history
    
    def get_best_epoch(self, metric: str = 'mse', phase: str = 'val') -> tuple:
        """
        Get epoch with best (lowest) metric value.
        
        Args:
            metric: Metric name to evaluate
            phase: 'train' or 'val'
            
        Returns:
            Tuple of (best_epoch, best_value)
        """
        if not self.history[phase][metric]:
            return (0, float('inf'))
        
        values = self.history[phase][metric]
        best_idx = np.argmin(values)
        best_epoch = self.history['epochs'][best_idx]
        best_value = values[best_idx]
        
        return (best_epoch, best_value)
    
    def save_history(self, path: str):
        """
        Save training history to JSON file.
        
        Args:
            path: Path to save history file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self, path: str):
        """
        Load training history from JSON file.
        
        Args:
            path: Path to history file
        """
        with open(path, 'r') as f:
            self.history = json.load(f)
    
    def get_latest_metrics(self, phase: str = 'val') -> Optional[Dict[str, float]]:
        """
        Get most recent metrics for a phase.
        
        Args:
            phase: 'train' or 'val'
            
        Returns:
            Dictionary of latest metrics or None if no history
        """
        if not self.history['epochs']:
            return None
        
        latest_metrics = {}
        for metric_name in self.metrics_list:
            if self.history[phase][metric_name]:
                latest_metrics[metric_name] = self.history[phase][metric_name][-1]
        
        return latest_metrics if latest_metrics else None

