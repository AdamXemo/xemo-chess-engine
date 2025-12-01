"""
Example usage of the chess neural network data loading pipeline.

This script demonstrates how to use the implemented functionality
for loading and preprocessing chess data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data import ChessDataLoader, BitboardConverter
from src.config import ExperimentConfig, DataConfig, TrainingConfig


def example_basic_loading():
    """Example 1: Basic data loading from CSV."""
    print("=" * 60)
    print("Example 1: Basic Data Loading")
    print("=" * 60)
    
    # Load raw data
    fen_list, eval_list = ChessDataLoader.load_from_csv(
        'data/chess_data.csv',
        max_samples=1000
    )
    
    print(f"Loaded {len(fen_list)} positions")
    print(f"Sample position: {fen_list[0]}")
    print(f"Sample evaluation: {eval_list[0]}")


def example_bitboard_conversion():
    """Example 2: Converting FEN to bitboard."""
    print("\n" + "=" * 60)
    print("Example 2: Bitboard Conversion")
    print("=" * 60)
    
    # Starting position
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Convert to different representations
    simple_board = BitboardConverter.convert_fen(starting_fen, 'simple')
    full_board = BitboardConverter.convert_fen(starting_fen, 'full')
    
    print(f"Simple representation shape: {simple_board.shape}")
    print(f"Full representation shape: {full_board.shape}")


def example_dataset_creation():
    """Example 3: Creating PyTorch datasets."""
    print("\n" + "=" * 60)
    print("Example 3: Creating PyTorch Datasets")
    print("=" * 60)
    
    # Create train/val/test datasets
    train_dataset, val_dataset, test_dataset = ChessDataLoader.create_datasets(
        csv_path='data/chess_data.csv',
        representation='full',
        max_samples=10000,  # Use 10k samples for quick testing
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        normalize_eval=False,
        random_seed=42
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Get a sample
    bitboard, evaluation = train_dataset[0]
    print(f"\nSample bitboard shape: {bitboard.shape}")
    print(f"Sample evaluation: {evaluation.item():.3f}")


def example_dataloader():
    """Example 4: Using PyTorch DataLoader."""
    print("\n" + "=" * 60)
    print("Example 4: PyTorch DataLoader")
    print("=" * 60)
    
    # Create dataset
    train_dataset, _, _ = ChessDataLoader.create_datasets(
        csv_path='data/chess_data.csv',
        representation='full',
        max_samples=1000,
        random_seed=42
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Number of batches: {len(train_loader)}")
    
    # Iterate through a few batches
    for i, (bitboards, evaluations) in enumerate(train_loader):
        print(f"\nBatch {i+1}:")
        print(f"  Bitboards shape: {bitboards.shape}")
        print(f"  Evaluations shape: {evaluations.shape}")
        print(f"  Evaluation range: [{evaluations.min():.2f}, {evaluations.max():.2f}]")
        
        if i >= 2:  # Just show first 3 batches
            break


def example_configuration():
    """Example 5: Using configuration system."""
    print("\n" + "=" * 60)
    print("Example 5: Configuration System")
    print("=" * 60)
    
    # Create a configuration
    config = ExperimentConfig(
        experiment_name="example_experiment",
        description="Testing configuration system",
        tags=["example", "test"],
        data=DataConfig(
            csv_path="data/chess_data.csv",
            representation='full',
            max_samples=10000,
            normalize_eval=False
        ),
        training=TrainingConfig(
            batch_size=256,
            num_epochs=50,
            learning_rate=0.001
        )
    )
    
    print(f"Experiment: {config.experiment_name}")
    print(f"Representation: {config.data.representation}")
    print(f"Input channels: {config.model.input_channels}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Save configuration
    config.save('experiments/configs/example_config.json')
    print("\nConfiguration saved to experiments/configs/example_config.json")


def example_multiple_representations():
    """Example 6: Comparing different representations."""
    print("\n" + "=" * 60)
    print("Example 6: Comparing Representations")
    print("=" * 60)
    
    representations = ['simple', 'essential', 'full']
    
    for rep in representations:
        dataset, _, _ = ChessDataLoader.create_datasets(
            csv_path='data/chess_data.csv',
            representation=rep,
            max_samples=100,
            random_seed=42
        )
        
        bitboard, _ = dataset[0]
        print(f"\n{rep.capitalize()} representation:")
        print(f"  Shape: {bitboard.shape}")
        print(f"  Memory per sample: {bitboard.element_size() * bitboard.nelement()} bytes")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Chess Neural Network - Example Usage")
    print("=" * 60)
    
    # Run all examples
    example_basic_loading()
    example_bitboard_conversion()
    example_dataset_creation()
    example_dataloader()
    example_configuration()
    example_multiple_representations()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

