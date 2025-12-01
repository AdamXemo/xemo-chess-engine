"""
Training demo showing how to train a chess evaluation model.

Demonstrates the complete training pipeline with a small dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models import ChessCNN
from src.data import ChessDataLoader
from src.training import Trainer
from src.config import ExperimentConfig, DataConfig, TrainingConfig, CNNConfig
from src.utils import print_header, print_success, print_info


def main():
    """Run training demo."""
    print_header("Chess CNN Training Demo")
    
    # Configuration
    print_info("Setting up configuration...")
    config = ExperimentConfig(
        experiment_name="training_demo",
        description="Quick training demo with small dataset",
        tags=["demo", "cnn", "small"],
        data=DataConfig(
            csv_path='data/chess_data.csv',
            representation='full',
            max_samples=10000,  # Small dataset for demo
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        ),
        model=CNNConfig(
            input_channels=23,
            base_channels=32,  # Smaller model for faster training
            dropout_conv=0.3,
            dropout_fc=0.5
        ),
        training=TrainingConfig(
            batch_size=128,
            num_epochs=10,  # Just 10 epochs for demo
            learning_rate=0.001,
            optimizer='adam',
            use_scheduler=True,
            scheduler_type='reduce_on_plateau',
            scheduler_patience=3,
            loss_function='mse',
            gradient_clip=1.0,
            use_early_stopping=True,
            early_stopping_patience=5,
            save_every_epoch=False,  # Only save best for demo
            validate_every=1,
            log_interval=10
        ),
        random_seed=42
    )
    
    print_success("Configuration created")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Dataset size: {config.data.max_samples:,} samples")
    print(f"  Model: CNN with {config.model.base_channels} base channels")
    print(f"  Epochs: {config.training.num_epochs}")
    print()
    
    # Load data
    print_info("Loading dataset...")
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
    
    print_success(f"Data loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
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
    print_info("Setting up trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    print_success(f"Trainer ready on device: {device}")
    print()
    
    # Train
    print_header("Starting Training")
    trainer.train()
    
    # Evaluate on test set
    print()
    print_header("Test Evaluation")
    test_metrics = trainer.evaluate(test_loader)
    
    # Show results
    print()
    print_success("Training demo complete!")
    print_info(f"Checkpoints saved to: experiments/checkpoints/{config.experiment_name}/")
    print_info(f"Results saved to: experiments/results/{config.experiment_name}/")
    print_info(f"Logs saved to: experiments/logs/")


if __name__ == '__main__':
    main()

