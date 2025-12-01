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


def test_csv_loading():
    """Test CSV loading functionality."""
    print("=" * 60)
    print("Testing CSV Loading")
    print("=" * 60)
    
    # Load a small sample
    fen_list, eval_list = ChessDataLoader.load_from_csv(
        'data/chess_data.csv',
        max_samples=100
    )
    
    print(f"\nLoaded {len(fen_list)} positions")
    print(f"\nFirst 3 samples:")
    for i in range(min(3, len(fen_list))):
        print(f"  {i+1}. FEN: {fen_list[i][:50]}...")
        print(f"     Eval: {eval_list[i]}")
    
    # Check evaluation range
    min_eval = min(eval_list)
    max_eval = max(eval_list)
    print(f"\nEvaluation range: [{min_eval:.2f}, {max_eval:.2f}]")
    
    return fen_list, eval_list


def test_bitboard_conversion(fen_list):
    """Test bitboard conversion for all representations."""
    print("\n" + "=" * 60)
    print("Testing Bitboard Conversion")
    print("=" * 60)
    
    representations = ['simple', 'essential', 'full']
    expected_channels = {'simple': 13, 'essential': 19, 'full': 23}
    
    test_fen = fen_list[0]
    print(f"\nTest FEN: {test_fen}")
    
    for rep in representations:
        bitboard = BitboardConverter.convert_fen(test_fen, representation=rep)
        expected = expected_channels[rep]
        
        print(f"\n{rep.capitalize()} representation:")
        print(f"  Shape: {bitboard.shape}")
        print(f"  Expected channels: {expected}")
        print(f"  Data type: {bitboard.dtype}")
        print(f"  Value range: [{bitboard.min()}, {bitboard.max()}]")
        print(f"  Non-zero elements: {np.count_nonzero(bitboard)}")
        
        assert bitboard.shape == (expected, 8, 8), \
            f"Expected shape ({expected}, 8, 8), got {bitboard.shape}"
        assert bitboard.dtype == np.uint8, \
            f"Expected dtype uint8, got {bitboard.dtype}"
        
        print(f"  ✓ {rep.capitalize()} conversion passed!")


def test_dataset_and_dataloader():
    """Test PyTorch Dataset and DataLoader integration."""
    print("\n" + "=" * 60)
    print("Testing PyTorch Dataset and DataLoader")
    print("=" * 60)
    
    # Create datasets
    print("\nCreating datasets with 1000 samples...")
    train_dataset, val_dataset, test_dataset = ChessDataLoader.create_datasets(
        csv_path='data/chess_data.csv',
        representation='full',
        max_samples=1000,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize_eval=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Test getting a single item
    print("\nTesting single item retrieval:")
    bitboard, evaluation = train_dataset[0]
    print(f"  Bitboard shape: {bitboard.shape}")
    print(f"  Bitboard dtype: {bitboard.dtype}")
    print(f"  Evaluation: {evaluation.item():.3f}")
    print(f"  Evaluation shape: {evaluation.shape}")
    
    # Test DataLoader
    print("\nTesting DataLoader:")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    # Get one batch
    batch_bitboards, batch_evals = next(iter(train_loader))
    print(f"  Batch bitboards shape: {batch_bitboards.shape}")
    print(f"  Batch evaluations shape: {batch_evals.shape}")
    print(f"  Batch size: {batch_bitboards.shape[0]}")
    
    assert batch_bitboards.shape == (32, 23, 8, 8), \
        f"Expected batch shape (32, 23, 8, 8), got {batch_bitboards.shape}"
    assert batch_evals.shape == (32, 1), \
        f"Expected evaluation shape (32, 1), got {batch_evals.shape}"
    
    print("\n  ✓ DataLoader integration passed!")
    
    # Test all representations
    print("\nTesting all representations with DataLoader:")
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
        
        print(f"  {rep}: {batch_bb.shape} ✓")
        assert batch_bb.shape[1] == channels


def test_normalized_evaluations():
    """Test evaluation normalization."""
    print("\n" + "=" * 60)
    print("Testing Evaluation Normalization")
    print("=" * 60)
    
    # Without normalization
    dataset_raw, _, _ = ChessDataLoader.create_datasets(
        csv_path='data/chess_data.csv',
        representation='simple',
        max_samples=100,
        normalize_eval=False
    )
    
    # With normalization
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
    
    print(f"\nRaw evaluation: {eval_raw.item():.3f}")
    print(f"Normalized evaluation: {eval_norm.item():.3f}")
    print(f"Expected normalized: {eval_raw.item() / 100.0:.3f}")
    
    assert abs(eval_norm.item() - eval_raw.item() / 100.0) < 1e-5, \
        "Normalization not working correctly"
    
    print("\n  ✓ Normalization test passed!")


if __name__ == '__main__':
    try:
        import numpy as np
        
        print("\n" + "=" * 60)
        print("Chess Neural Network - Data Loading Test Suite")
        print("=" * 60)
        
        # Run tests
        fen_list, eval_list = test_csv_loading()
        test_bitboard_conversion(fen_list)
        test_dataset_and_dataloader()
        test_normalized_evaluations()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nData loading and preprocessing pipeline is working correctly.")
        print("You can now proceed with model implementation and training.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

