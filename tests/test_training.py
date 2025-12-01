"""
Tests for training components.

Tests metrics computation, early stopping, and training functionality.
"""

import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    MetricsTracker,
    EarlyStopping,
    compute_mse,
    compute_mae,
    compute_rmse,
    set_seed,
    get_lr,
)
from src.models import ChessCNN
from src.data import ChessDataLoader
from src.config import ExperimentConfig, DataConfig, TrainingConfig, CNNConfig
from src.training import Trainer
from src.utils import print_header, print_section, print_success, print_info


def test_metrics_computation():
    """Test metric computation functions."""
    print_section("Testing Metrics Computation")
    
    # Create dummy predictions and targets
    predictions = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    targets = torch.tensor([[1.5], [2.5], [2.5], [3.5]])
    
    # Test MSE
    mse = compute_mse(predictions, targets)
    print_info(f"MSE: {mse:.4f}")
    assert mse > 0, "MSE should be positive"
    
    # Test MAE
    mae = compute_mae(predictions, targets)
    print_info(f"MAE: {mae:.4f}")
    assert mae > 0, "MAE should be positive"
    
    # Test RMSE
    rmse = compute_rmse(predictions, targets)
    print_info(f"RMSE: {rmse:.4f}")
    assert rmse > 0, "RMSE should be positive"
    
    # RMSE should be sqrt of MSE
    assert abs(rmse - (mse ** 0.5)) < 1e-5, "RMSE should equal sqrt(MSE)"
    
    print_success("Metrics computation tests passed!")


def test_metrics_tracker():
    """Test MetricsTracker class."""
    print_section("Testing MetricsTracker")
    
    tracker = MetricsTracker(metrics_list=['mse', 'mae'])
    
    # Reset
    tracker.reset()
    assert tracker.num_samples == 0
    
    # Add some batches
    for i in range(3):
        pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        tracker.update(pred, target)
    
    assert tracker.num_samples == 12, "Should have 12 samples total"
    
    # Compute metrics
    metrics = tracker.compute()
    assert 'mse' in metrics
    assert 'mae' in metrics
    print_info(f"Computed metrics: {metrics}")
    
    # Add to history
    tracker.add_to_history(metrics, epoch=0, phase='train')
    assert len(tracker.history['train']['mse']) == 1
    assert len(tracker.history['epochs']) == 1
    
    # Save and load history
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    tracker.save_history(temp_path)
    
    new_tracker = MetricsTracker()
    new_tracker.load_history(temp_path)
    assert len(new_tracker.history['epochs']) == 1
    
    Path(temp_path).unlink()  # Clean up
    
    print_success("MetricsTracker tests passed!")


def test_early_stopping():
    """Test EarlyStopping class."""
    print_section("Testing Early Stopping")
    
    # Test with min mode (for loss)
    early_stop = EarlyStopping(patience=3, min_delta=0.01, mode='min')
    
    # Sequence of losses
    losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]
    
    should_stop = False
    for i, loss in enumerate(losses):
        should_stop = early_stop(loss)
        if should_stop:
            print_info(f"Early stopping triggered at iteration {i}")
            break
    
    assert should_stop, "Early stopping should have triggered"
    assert early_stop.counter >= early_stop.patience
    
    # Test reset
    early_stop.reset()
    assert early_stop.counter == 0
    assert not early_stop.early_stop
    
    print_success("Early stopping tests passed!")


def test_training_utilities():
    """Test training utility functions."""
    print_section("Testing Training Utilities")
    
    # Test set_seed
    set_seed(42)
    val1 = torch.rand(1).item()
    set_seed(42)
    val2 = torch.rand(1).item()
    assert val1 == val2, "Setting seed should produce same random values"
    print_success("set_seed works!")
    
    # Test get_lr
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr = get_lr(optimizer)
    assert abs(lr - 0.001) < 1e-9, "Should get correct learning rate"
    print_info(f"Learning rate: {lr}")
    print_success("get_lr works!")
    
    print_success("Training utilities tests passed!")


def test_single_epoch_training():
    """Test single epoch of training."""
    print_section("Testing Single Epoch Training")
    
    # Create small dummy dataset
    print_info("Creating dummy dataset...")
    X = torch.randn(100, 23, 8, 8)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    # Create small model
    print_info("Creating small model...")
    model = ChessCNN(input_channels=23, base_channels=16)
    
    # Create simple config
    config = ExperimentConfig(
        experiment_name="test_training",
        data=DataConfig(max_samples=100),
        model=CNNConfig(base_channels=16),
        training=TrainingConfig(
            batch_size=10,
            num_epochs=2,
            learning_rate=0.01,
            use_early_stopping=False,
            save_every_epoch=False
        )
    )
    
    # Create trainer
    print_info("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu'
    )
    
    # Train for 1 epoch manually
    print_info("Training for 2 epochs...")
    trainer.train()
    
    # Check that metrics were tracked
    history = trainer.metrics_tracker.get_history()
    assert len(history['epochs']) == 2, "Should have 2 epochs of history"
    assert len(history['train']['mse']) == 2
    assert len(history['val']['mse']) == 2
    
    print_success("Single epoch training test passed!")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print_section("Testing Checkpoint Save/Load")
    
    # Create small model and dataset
    X = torch.randn(50, 23, 8, 8)
    y = torch.randn(50, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)
    
    model = ChessCNN(input_channels=23, base_channels=16)
    
    config = ExperimentConfig(
        experiment_name="test_checkpoint",
        training=TrainingConfig(
            batch_size=10,
            num_epochs=1,
            save_every_epoch=True
        )
    )
    
    trainer = Trainer(model, loader, loader, config, device='cpu')
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.checkpoint_dir = Path(tmpdir)
        trainer.save_checkpoint(epoch=0, is_best=True)
        
        checkpoint_path = Path(tmpdir) / "best_model.pth"
        assert checkpoint_path.exists(), "Checkpoint should be saved"
        
        # Load checkpoint
        trainer2 = Trainer(model, loader, loader, config, device='cpu')
        trainer2.load_checkpoint(str(checkpoint_path))
        
        assert trainer2.start_epoch == 1, "Should resume from next epoch"
    
    print_success("Checkpoint save/load test passed!")


if __name__ == '__main__':
    print_header("Training Components Test Suite")
    
    try:
        test_metrics_computation()
        test_metrics_tracker()
        test_early_stopping()
        test_training_utilities()
        test_single_epoch_training()
        test_checkpoint_save_load()
        
        print()
        print_header("All Training Tests Passed!")
        print_success("Training pipeline is ready to use")
        
    except Exception as e:
        from src.utils import print_error
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

