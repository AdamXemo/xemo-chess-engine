"""
Main Trainer class for model training.

Handles the complete training pipeline including training loops,
validation, checkpointing, early stopping, and logging.
"""

import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..utils import ExperimentLogger, TrainingProgress
from .metrics import MetricsTracker
from .early_stopping import EarlyStopping
from .utils import set_seed, get_lr, save_training_curves, format_time


class Trainer:
    """
    Main trainer class for chess position evaluation models.
    
    Handles complete training pipeline with logging, checkpointing,
    and early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ExperimentConfig,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Experiment configuration
            device: Device to train on ('cuda' or 'cpu'), auto-detect if None
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_loss_function()
        
        # Setup metrics tracker
        self.metrics_tracker = MetricsTracker(metrics_list=['mse', 'mae', 'rmse'])
        
        # Setup early stopping
        if config.training.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta
            )
        else:
            self.early_stopping = None
        
        # Setup logger
        self.logger = ExperimentLogger()
        self.logger.setup_experiment(
            config.experiment_name,
            log_dir=config.training.log_dir
        )
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.training_start_time = None
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir) / config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup optimizer based on configuration.
        
        Returns:
            Configured optimizer
        """
        cfg = self.config.training
        
        if cfg.optimizer.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.learning_rate,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Setup learning rate scheduler.
        
        Returns:
            Configured scheduler or None
        """
        if not self.config.training.use_scheduler:
            return None
        
        cfg = self.config.training
        
        if cfg.scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=cfg.scheduler_factor,
                patience=cfg.scheduler_patience,
                min_lr=cfg.min_lr
            )
        elif cfg.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.num_epochs,
                eta_min=cfg.min_lr
            )
        elif cfg.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            self.logger.warning(f"Unknown scheduler type: {cfg.scheduler_type}")
            return None
    
    def _setup_loss_function(self) -> nn.Module:
        """
        Setup loss function.
        
        Returns:
            Loss function
        """
        loss_name = self.config.training.loss_function.lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae' or loss_name == 'l1':
            return nn.L1Loss()
        elif loss_name == 'huber':
            return nn.HuberLoss(delta=self.config.training.huber_delta)
        else:
            self.logger.warning(f"Unknown loss function: {loss_name}, using MSE")
            return nn.MSELoss()
    
    def train(self):
        """
        Main training loop.
        
        Trains the model for the specified number of epochs with validation,
        checkpointing, and early stopping.
        """
        # Set random seed
        set_seed(self.config.random_seed)
        
        # Log training start
        self.logger.header("Training Started")
        self.logger.config(self.config.to_dict())
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        # Training time tracking
        self.training_start_time = time.time()
        
        # Main training loop with progress bar
        with TrainingProgress() as progress:
            progress.create_epoch_bar(self.config.training.num_epochs, "Training")
            
            for epoch in range(self.start_epoch, self.config.training.num_epochs):
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self.train_epoch(epoch, progress)
                
                # Validation phase
                if epoch % self.config.training.validate_every == 0:
                    val_metrics = self.validate_epoch(epoch)
                else:
                    val_metrics = None
                
                # Update learning rate scheduler
                if self.scheduler is not None and val_metrics is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['mse'])
                    else:
                        self.scheduler.step()
                
                # Log epoch metrics
                epoch_time = time.time() - epoch_start_time
                self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)
                
                # Save checkpoint
                is_best = False
                if val_metrics is not None:
                    val_loss = val_metrics['mse']
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        is_best = True
                
                if self.config.training.save_every_epoch or is_best:
                    self.save_checkpoint(epoch, is_best=is_best)
                
                # Check early stopping
                if self.early_stopping is not None and val_metrics is not None:
                    if self.early_stopping(val_metrics['mse']):
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
                        break
                
                # Update epoch progress
                progress.update_epoch(1)
        
        # Training complete
        total_time = time.time() - self.training_start_time
        self._finalize_training(total_time)
    
    def train_epoch(self, epoch: int, progress: TrainingProgress) -> dict:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            progress: Progress bar manager
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.metrics_tracker.reset()
        
        # Create batch progress bar
        progress.create_batch_bar(len(self.train_loader), f"Epoch {epoch + 1}")
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            # Update metrics
            self.metrics_tracker.update(outputs, targets)
            
            # Update progress bar with live metrics
            if batch_idx % self.config.training.log_interval == 0:
                current_metrics = {
                    'loss': loss.item(),
                    'lr': get_lr(self.optimizer)
                }
                progress.update_metrics(current_metrics)
            
            progress.update_batch(1)
        
        # Compute epoch metrics
        train_metrics = self.metrics_tracker.compute()
        self.metrics_tracker.add_to_history(train_metrics, epoch, phase='train')
        
        return train_metrics
    
    def validate_epoch(self, epoch: int) -> dict:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Update metrics
                self.metrics_tracker.update(outputs, targets)
        
        # Compute epoch metrics
        val_metrics = self.metrics_tracker.compute()
        self.metrics_tracker.add_to_history(val_metrics, epoch, phase='val')
        
        return val_metrics
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of test metrics
        """
        self.logger.info("Evaluating on test set...")
        
        self.model.eval()
        metrics_tracker = MetricsTracker(metrics_list=['mse', 'mae', 'rmse'])
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                metrics_tracker.update(outputs, targets)
        
        test_metrics = metrics_tracker.compute()
        
        self.logger.success("Test evaluation complete")
        self.logger.table(test_metrics, title="Test Metrics")
        
        return test_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_tracker.get_history(),
            'config': self.config.to_dict(),
        }
        
        if self.early_stopping is not None:
            checkpoint['early_stopping_state'] = self.early_stopping.state_dict()
        
        # Save regular checkpoint
        if self.config.training.save_every_epoch:
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
        
        # Save last checkpoint
        last_checkpoint_path = self.checkpoint_dir / "last_checkpoint.pth"
        torch.save(checkpoint, last_checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            self.logger.success(f"New best model saved! Val loss: {self.best_val_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'metrics_history' in checkpoint:
            self.metrics_tracker.history = checkpoint['metrics_history']
        
        if self.early_stopping is not None and 'early_stopping_state' in checkpoint:
            self.early_stopping.load_state_dict(checkpoint['early_stopping_state'])
        
        self.logger.success(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")
    
    def _log_epoch(self, epoch: int, train_metrics: dict, val_metrics: Optional[dict], epoch_time: float):
        """
        Log epoch metrics.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics (can be None)
            epoch_time: Time taken for epoch
        """
        # Prepare metrics for logging
        log_metrics = {
            'Epoch': epoch + 1,
            'Train Loss': f"{train_metrics['mse']:.6f}",
            'Train MAE': f"{train_metrics['mae']:.6f}",
            'LR': f"{get_lr(self.optimizer):.2e}",
            'Time': format_time(epoch_time)
        }
        
        if val_metrics is not None:
            log_metrics['Val Loss'] = f"{val_metrics['mse']:.6f}"
            log_metrics['Val MAE'] = f"{val_metrics['mae']:.6f}"
        
        self.logger.table(log_metrics, title=f"Epoch {epoch + 1}")
    
    def _finalize_training(self, total_time: float):
        """
        Finalize training and save results.
        
        Args:
            total_time: Total training time
        """
        self.logger.success("Training Complete!")
        
        # Get best epoch info
        best_epoch, best_val_loss = self.metrics_tracker.get_best_epoch(metric='mse', phase='val')
        
        # Log summary
        summary = {
            'Total Epochs': len(self.metrics_tracker.history['epochs']),
            'Best Epoch': best_epoch,
            'Best Val Loss': f"{best_val_loss:.6f}",
            'Total Time': format_time(total_time),
            'Final LR': f"{get_lr(self.optimizer):.2e}"
        }
        self.logger.table(summary, title="Training Summary")
        
        # Save training history
        results_dir = Path('experiments/results') / self.config.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        history_path = results_dir / 'training_history.json'
        self.metrics_tracker.save_history(history_path)
        self.logger.info(f"Training history saved to: {history_path}")
        
        # Save training curves
        curves_path = results_dir / 'training_curves.png'
        save_training_curves(
            self.metrics_tracker.get_history(),
            curves_path,
            metrics=['mse', 'mae']
        )
        self.logger.info(f"Training curves saved to: {curves_path}")
        
        # Close logger
        self.logger.close()
    
    def save_best_to_folder(
        self,
        model_type: str = 'cnn',
        save_dir: str = 'best_models',
        training_time: float = 0.0
    ) -> tuple:
        """
        Save best model with metadata to dedicated folder.
        
        Args:
            model_type: Type of model ('cnn', 'resnet', etc.)
            save_dir: Directory to save best model
            training_time: Total training time in seconds
            
        Returns:
            Tuple of (model_path, yaml_path)
        """
        from ..utils.model_io import save_best_model
        
        # Get metrics
        history = self.metrics_tracker.get_history()
        best_epoch, best_val_loss = self.metrics_tracker.get_best_epoch(metric='mse', phase='val')
        
        # Get MAE for best epoch
        best_val_mae = 0.0
        if history['val']['mae']:
            best_idx = history['epochs'].index(best_epoch) if best_epoch in history['epochs'] else -1
            if best_idx >= 0 and best_idx < len(history['val']['mae']):
                best_val_mae = history['val']['mae'][best_idx]
        
        # Get final metrics
        final_train_loss = history['train']['mse'][-1] if history['train']['mse'] else 0.0
        final_val_loss = history['val']['mse'][-1] if history['val']['mse'] else 0.0
        
        metrics = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'best_val_mae': best_val_mae,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'epochs_trained': len(history['epochs']),
        }
        
        # Dataset info
        dataset_info = {
            'train': len(self.train_loader.dataset),
            'val': len(self.val_loader.dataset),
            'test': 0,  # Will be updated if test loader provided
        }
        
        # Save
        model_path, yaml_path = save_best_model(
            model=self.model,
            model_type=model_type,
            config=self.config,
            metrics=metrics,
            training_time=training_time,
            dataset_info=dataset_info,
            save_dir=save_dir,
            device=self.device
        )
        
        self.logger.success(f"Best model saved to: {model_path}")
        self.logger.info(f"Metadata saved to: {yaml_path}")
        
        return model_path, yaml_path

