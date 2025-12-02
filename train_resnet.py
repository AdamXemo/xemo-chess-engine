#!/usr/bin/env python3
"""
Train ResNet model for chess position evaluation.

This script trains the ChessResNet model on the full dataset for 6-8 hours.
Based on AlphaZero architecture with residual blocks and tanh output.

Usage:
    python train_resnet.py

Output:
    - best_models/resnet_{date}_best.pth  - Best model weights
    - best_models/resnet_{date}_best.yaml - Full training metadata
    - experiments/checkpoints/{experiment_name}/ - Training checkpoints
    - experiments/results/{experiment_name}/ - Training history and curves
    - experiments/logs/{experiment_name}_{timestamp}.log - Training log
"""

import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from src.models import ChessResNet
from src.data import ChessDataLoader
from src.training import Trainer
from src.config import ExperimentConfig, DataConfig, TrainingConfig, ResNetConfig
from src.utils import print_header, print_success, print_info, print_warning, print_metric_table


def main():
    """Run full ResNet training."""
    
    # Generate experiment name with date
    date_str = datetime.now().strftime("%Y%m%d")
    experiment_name = f"resnet_full_training_{date_str}"
    
    print_header("Chess ResNet Training")
    print_info(f"Experiment: {experiment_name}")
    print_info(f"Target training time: ~6 hours")
    print_info(f"Architecture: 15 ResBlocks, 192 filters, AlphaZero value head")
    print()
    
    # Configuration for 6-8 hour training on RTX 4060
    # Based on research recommendations
    config = ExperimentConfig(
        experiment_name=experiment_name,
        description="ResNet training with AlphaZero-style architecture",
        tags=["resnet", "full", "production", "alphazero"],
        
        data=DataConfig(
            csv_path='data/chess_data.csv',
            representation='full',  # 23 channels
            max_samples=None,  # Use all 2.2M samples
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            normalize_eval=True,  # Normalize to [-1, 1] for tanh output
            random_seed=42
        ),
        
        model=ResNetConfig(
            input_channels=23,
            num_blocks=15,      # 15 residual blocks (deeper)
            num_filters=192,    # 192 filters per layer (wider)
            value_head_hidden=256,  # AlphaZero style
        ),
        
        training=TrainingConfig(
            # Optimization (based on research)
            batch_size=256,      # Smaller batch for larger model (memory)
            num_epochs=50,       # More epochs, early stopping will handle it
            learning_rate=0.001,
            weight_decay=0.0001,  # L2 regularization
            optimizer='adamw',    # AdamW recommended by research
            
            # Learning rate scheduling (cosine annealing recommended)
            use_scheduler=True,
            scheduler_type='cosine',  # Better than step decay
            min_lr=1e-6,
            
            # Loss function
            loss_function='mse',
            
            # Regularization
            gradient_clip=1.0,  # Stability for deeper network
            
            # Checkpointing
            save_every_epoch=False,  # Only save best to save disk space
            checkpoint_dir='experiments/checkpoints',
            
            # Logging
            log_dir='experiments/logs',
            log_interval=100,  # Log every 100 batches
            
            # Validation
            validate_every=1,
            
            # Early stopping (longer patience for larger model)
            use_early_stopping=True,
            early_stopping_patience=15,  # More patience for bigger model
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
        print_warning("CUDA not available, training on CPU (will be VERY slow!)")
        print_warning("Consider reducing max_samples for testing")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_success(f"Using GPU: {gpu_name}")
        print_info(f"GPU Memory: {gpu_mem:.1f} GB")
    print()
    
    # Model summary
    print_info("Model Configuration:")
    model_info = {
        'Architecture': 'ChessResNet (AlphaZero-style)',
        'Residual Blocks': config.model.num_blocks,
        'Filters': config.model.num_filters,
        'Value Head': f'1x1 conv → FC {config.model.value_head_hidden} → tanh',
        'Output Range': '[-1, 1]',
    }
    print_metric_table(model_info, title="ResNet Configuration")
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
        normalize_eval=config.data.normalize_eval,  # Normalize for tanh
        random_seed=config.data.random_seed
    )
    
    load_time = time.time() - start_load
    print_success(f"Data loaded in {load_time:.1f}s")
    print_info(f"  Train: {len(train_ds):,} samples")
    print_info(f"  Val:   {len(val_ds):,} samples")
    print_info(f"  Test:  {len(test_ds):,} samples")
    print_info(f"  Normalization: {'[-1, 1]' if config.data.normalize_eval else '[-100, 100]'}")
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
    
    batches_per_epoch = len(train_loader)
    print_info(f"Batches per epoch: {batches_per_epoch:,}")
    print_info(f"Estimated time per epoch: ~7-8 min (RTX 4060)")
    print()
    
    # Create model
    print_info("Creating model...")
    model = ChessResNet(
        input_channels=config.model.input_channels,
        num_blocks=config.model.num_blocks,
        num_filters=config.model.num_filters,
        value_head_hidden=config.model.value_head_hidden
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
    print_info("This will take approximately 6-8 hours...")
    print_info("Press Ctrl+C to stop early (checkpoints will be saved)")
    print()
    
    training_start = time.time()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print()
        print_warning("Training interrupted by user")
        print_info("Saving current state...")
    
    training_time = time.time() - training_start
    
    # Save best model to best_models folder
    print()
    print_header("Saving Best Model")
    
    model_path, yaml_path = trainer.save_best_to_folder(
        model_type='resnet',
        save_dir='best_models',
        training_time=training_time
    )
    
    # Evaluate on test set
    print()
    print_header("Test Set Evaluation")
    
    # Load best model for evaluation
    best_checkpoint_path = trainer.checkpoint_dir / "best_model.pth"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = trainer.evaluate(test_loader)
    
    # Final summary
    print()
    print_header("Training Complete!")
    print_success(f"Best model saved to: {model_path}")
    print_success(f"Metadata saved to: {yaml_path}")
    print()
    
    # Results summary
    print_info("Results summary:")
    results = {
        'Test MSE': f"{test_metrics['mse']:.6f}",
        'Test MAE': f"{test_metrics['mae']:.6f}",
        'Test RMSE': f"{test_metrics['rmse']:.6f}",
    }
    print_metric_table(results, title="Test Metrics")
    print()
    
    # Note about output scale
    if config.data.normalize_eval:
        print_info("Note: Metrics are in normalized scale [-1, 1]")
        print_info("To convert to centipawns, multiply by 100")
        print_info(f"  → Test MAE ≈ {test_metrics['mae'] * 100:.2f} centipawns")
    
    # Format training time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print()
    print_info(f"Total training time: {hours}h {minutes}m {seconds}s")
    
    print()
    print_success("All done! Your ResNet model is ready to use.")
    print_info("Load it with:")
    print(f"  from src.utils import load_best_model")
    print(f"  from src.models import ChessResNet")
    print(f"  model, info = load_best_model('{model_path}', ChessResNet)")


if __name__ == '__main__':
    main()

