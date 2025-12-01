"""
Quick test script to verify data loading and bitboard conversion functionality.

This script loads a small sample of data and verifies that:
1. CSV parsing works correctly
2. FEN to bitboard conversion works
3. PyTorch DataLoader integration works
4. All three representations produce correct shapes
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data import ChessDataLoader, BitboardConverter
from src.utils import (
    print_header,
    print_section,
    print_success,
    print_error,
    print_info,
    print_metric_table,
)


def test_csv_loading():
    """Test CSV loading functionality."""
    print_section("Testing CSV Loading")
    
    # Load a small sample
    print_info("Loading sample data from CSV...")
    fen_list, eval_list = ChessDataLoader.load_from_csv(
        'data/chess_data.csv',
        max_samples=100
    )
    
    print_success(f"Loaded {len(fen_list)} positions")
    
    print_info("First 3 samples:")
    for i in range(min(3, len(fen_list))):
        print(f"  {i+1}. FEN: {fen_list[i][:50]}...")
        print(f"     Eval: {eval_list[i]}")
    
    # Check evaluation range
    min_eval = min(eval_list)
    max_eval = max(eval_list)
    stats = {
        'Total Samples': len(fen_list),
        'Min Evaluation': min_eval,
        'Max Evaluation': max_eval,
        'Eval Range': f"[{min_eval:.2f}, {max_eval:.2f}]"
    }
    print_metric_table(stats, title="Dataset Statistics")
    
    return fen_list, eval_list


def test_bitboard_conversion(fen_list):
    """Test bitboard conversion for all representations."""
    print_section("Testing Bitboard Conversion")
    
    representations = ['simple', 'essential', 'full']
    expected_channels = {'simple': 13, 'essential': 19, 'full': 23}
    
    test_fen = fen_list[0]
    print_info(f"Test FEN: {test_fen[:60]}...")
    
    print()
    for rep in representations:
        bitboard = BitboardConverter.convert_fen(test_fen, representation=rep)
        expected = expected_channels[rep]
        
        stats = {
            'Representation': rep.capitalize(),
            'Shape': str(bitboard.shape),
            'Expected Channels': expected,
            'Data Type': str(bitboard.dtype),
            'Value Range': f"[{bitboard.min()}, {bitboard.max()}]",
            'Non-zero Elements': np.count_nonzero(bitboard)
        }
        
        assert bitboard.shape == (expected, 8, 8), \
            f"Expected shape ({expected}, 8, 8), got {bitboard.shape}"
        assert bitboard.dtype == np.uint8, \
            f"Expected dtype uint8, got {bitboard.dtype}"
        
        print_success(f"{rep.capitalize()} conversion passed!")
        for key, val in stats.items():
            print(f"  {key}: {val}")
        print()


def test_dataset_and_dataloader():
    """Test PyTorch Dataset and DataLoader integration."""
    print_section("Testing PyTorch Dataset and DataLoader")
    
    # Create datasets
    print_info("Creating datasets with 1000 samples...")
    train_dataset, val_dataset, test_dataset = ChessDataLoader.create_datasets(
        csv_path='data/chess_data.csv',
        representation='full',
        max_samples=1000,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize_eval=False
    )
    
    dataset_stats = {
        'Train Samples': len(train_dataset),
        'Validation Samples': len(val_dataset),
        'Test Samples': len(test_dataset),
        'Total Samples': len(train_dataset) + len(val_dataset) + len(test_dataset)
    }
    print_metric_table(dataset_stats, title="Dataset Split")
    
    # Test getting a single item
    print_info("Testing single item retrieval...")
    bitboard, evaluation = train_dataset[0]
    item_stats = {
        'Bitboard Shape': str(bitboard.shape),
        'Bitboard Dtype': str(bitboard.dtype),
        'Evaluation': f"{evaluation.item():.3f}",
        'Evaluation Shape': str(evaluation.shape)
    }
    for key, val in item_stats.items():
        print(f"  {key}: {val}")
    
    # Test DataLoader
    print()
    print_info("Testing DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    # Get one batch
    batch_bitboards, batch_evals = next(iter(train_loader))
    batch_stats = {
        'Batch Bitboards Shape': str(batch_bitboards.shape),
        'Batch Evaluations Shape': str(batch_evals.shape),
        'Batch Size': batch_bitboards.shape[0]
    }
    
    assert batch_bitboards.shape == (32, 23, 8, 8), \
        f"Expected batch shape (32, 23, 8, 8), got {batch_bitboards.shape}"
    assert batch_evals.shape == (32, 1), \
        f"Expected evaluation shape (32, 1), got {batch_evals.shape}"
    
    print_success("DataLoader integration passed!")
    for key, val in batch_stats.items():
        print(f"  {key}: {val}")
    
    # Test all representations
    print()
    print_info("Testing all representations with DataLoader...")
    for rep in ['simple', 'essential', 'full']:
        channels = {'simple': 13, 'essential': 19, 'full': 23}[rep]
        
        dataset, _, _ = ChessDataLoader.create_datasets(
            csv_path='data/chess_data.csv',
            representation=rep,
            max_samples=100,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        loader = DataLoader(dataset, batch_size=16, num_workers=0)
        batch_bb, batch_ev = next(iter(loader))
        
        print_success(f"{rep}: {batch_bb.shape}")
        assert batch_bb.shape[1] == channels


def test_normalized_evaluations():
    """Test evaluation normalization."""
    print_section("Testing Evaluation Normalization")
    
    # Without normalization
    print_info("Testing without normalization...")
    dataset_raw, _, _ = ChessDataLoader.create_datasets(
        csv_path='data/chess_data.csv',
        representation='simple',
        max_samples=100,
        normalize_eval=False
    )
    
    # With normalization
    print_info("Testing with normalization...")
    dataset_norm, _, _ = ChessDataLoader.create_datasets(
        csv_path='data/chess_data.csv',
        representation='simple',
        max_samples=100,
        normalize_eval=True,
        random_seed=42  # Same seed to get same data
    )
    
    # Compare
    _, eval_raw = dataset_raw[0]
    _, eval_norm = dataset_norm[0]
    
    norm_stats = {
        'Raw Evaluation': f"{eval_raw.item():.3f}",
        'Normalized Evaluation': f"{eval_norm.item():.3f}",
        'Expected Normalized': f"{eval_raw.item() / 100.0:.3f}",
        'Difference': f"{abs(eval_norm.item() - eval_raw.item() / 100.0):.6f}"
    }
    print_metric_table(norm_stats, title="Normalization Test")
    
    assert abs(eval_norm.item() - eval_raw.item() / 100.0) < 1e-5, \
        "Normalization not working correctly"
    
    print_success("Normalization test passed!")


if __name__ == '__main__':
    try:
        import numpy as np
        
        print_header("Chess Neural Network - Data Loading Test Suite")
        
        # Run tests
        fen_list, eval_list = test_csv_loading()
        test_bitboard_conversion(fen_list)
        test_dataset_and_dataloader()
        test_normalized_evaluations()
        
        print()
        print_header("All Tests Passed!")
        print_success("Data loading and preprocessing pipeline is working correctly")
        print_info("You can now proceed with model implementation and training")
        
    except Exception as e:
        print_error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

