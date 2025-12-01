#!/usr/bin/env python3
"""
Train CNN model for chess position evaluation.

This script trains the ChessCNN model on the full dataset for approximately 1 hour.
It saves checkpoints, training curves, and the best model to the best_models/ folder.

Usage:
    python train_cnn.py

Output:
    - best_models/cnn_{date}_best.pth  - Best model weights
    - best_models/cnn_{date}_best.yaml - Full training metadata
    - experiments/checkpoints/{experiment_name}/ - Training checkpoints
    - experiments/results/{experiment_name}/ - Training history and curves
    - experiments/logs/{experiment_name}_{timestamp}.log - Training log
"""

import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.models import ChessCNN
from src.data import ChessDataLoader
from src.training import Trainer
from src.config import ExperimentConfig, DataConfig, TrainingConfig, CNNConfig
from src.utils import print_header, print_success, print_info, print_warning


def main():
    """Run full CNN training."""
    
    # Generate experiment name with date
    date_str = datetime.now().strftime("%Y%m%d")
    experiment_name = f"cnn_full_training_{date_str}"
    
    print_header("Chess CNN Training")
    print_info(f"Experiment: {experiment_name}")
    print_info(f"Target training time: ~1 hour")
    print()
    
    # Configuration optimized for ~1 hour training on RTX 4060
    config = ExperimentConfig(
        experiment_name=experiment_name,
        description="Full CNN training on chess evaluation dataset",
        tags=["cnn", "full", "production"],
        
        data=DataConfig(
            csv_path='data/chess_data.csv',
            representation='full',  # 23 channels
            max_samples=None,  # Use all 2.2M samples
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            normalize_eval=False,  # Keep [-100, 100] range
            random_seed=42
        ),
        
        model=CNNConfig(
            input_channels=23,
            base_channels=64,  # Full model
            dropout_conv=0.3,
            dropout_fc=0.5
        ),
        
        training=TrainingConfig(
            # Optimization
            batch_size=256,
            num_epochs=30,  # Approximately 1 hour with early stopping
            learning_rate=0.001,
            weight_decay=0.0001,
            optimizer='adam',
            
            # Learning rate scheduling
            use_scheduler=True,
            scheduler_type='reduce_on_plateau',
            scheduler_patience=3,
            scheduler_factor=0.5,
            min_lr=1e-6,
            
            # Loss function
            loss_function='mse',
            
            # Regularization
            gradient_clip=1.0,
            
            # Checkpointing
            save_every_epoch=False,  # Only save best to save disk space
            checkpoint_dir='experiments/checkpoints',
            
            # Logging
            log_dir='experiments/logs',
            log_interval=50,  # Log every 50 batches
            
            # Validation
            validate_every=1,
            
            # Early stopping
            use_early_stopping=True,
            early_stopping_patience=8,  # Stop if no improvement for 8 epochs
            early_stopping_min_delta=0.0001,
            
            # Data loading
            num_workers=4,
            pin_memory=True
        ),
        
        random_seed=42,
        deterministic=True
    )
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print_warning("CUDA not available, training on CPU (will be slow!)")
        print_warning("Consider reducing max_samples for testing")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print_success(f"Using GPU: {gpu_name}")
    print()
    
    # Load data
    print_info("Loading dataset...")
    start_load = time.time()
    
    train_ds, val_ds, test_ds = ChessDataLoader.create_datasets(
        csv_path=config.data.csv_path,
        representation=config.data.representation,
        max_samples=config.data.max_samples,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        normalize_eval=config.data.normalize_eval,
        random_seed=config.data.random_seed
    )
    
    load_time = time.time() - start_load
    print_success(f"Data loaded in {load_time:.1f}s")
    print_info(f"  Train: {len(train_ds):,} samples")
    print_info(f"  Val:   {len(val_ds):,} samples")
    print_info(f"  Test:  {len(test_ds):,} samples")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=True if config.training.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=True if config.training.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0  # Simpler for test
    )
    
    print_info(f"Batches per epoch: {len(train_loader):,}")
    print()
    
    # Create model
    print_info("Creating model...")
    model = ChessCNN(
        input_channels=config.model.input_channels,
        base_channels=config.model.base_channels,
        dropout_conv=config.model.dropout_conv,
        dropout_fc=config.model.dropout_fc
    )
    
    print_success(f"Model created: {model.count_parameters():,} parameters")
    print()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    print_header("Starting Training")
    training_start = time.time()
    
    trainer.train()
    
    training_time = time.time() - training_start
    
    # Save best model to best_models folder
    print()
    print_header("Saving Best Model")
    
    model_path, yaml_path = trainer.save_best_to_folder(
        model_type='cnn',
        save_dir='best_models',
        training_time=training_time
    )
    
    # Evaluate on test set
    print()
    print_header("Test Set Evaluation")
    
    # Load best model for evaluation
    best_checkpoint_path = trainer.checkpoint_dir / "best_model.pth"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = trainer.evaluate(test_loader)
    
    # Final summary
    print()
    print_header("Training Complete!")
    print_success(f"Best model saved to: {model_path}")
    print_success(f"Metadata saved to: {yaml_path}")
    print()
    print_info("Results summary:")
    print(f"  Test MSE:  {test_metrics['mse']:.6f}")
    print(f"  Test MAE:  {test_metrics['mae']:.6f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.6f}")
    print()
    
    # Format training time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print_info(f"Total training time: {hours}h {minutes}m {seconds}s")
    
    print()
    print_success("All done! Your model is ready to use.")
    print_info("Load it with:")
    print(f"  from src.utils import load_best_model")
    print(f"  from src.models import ChessCNN")
    print(f"  model, info = load_best_model('{model_path}', ChessCNN)")


if __name__ == '__main__':
    main()

