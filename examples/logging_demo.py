"""
Demo of logging and progress bar features.

Demonstrates all logging utilities, progress bars, and formatting.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    ExperimentLogger,
    get_logger,
    TrainingProgress,
    DataLoadProgress,
    SimpleProgress,
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_metric_table,
    print_config,
    print_comparison_table,
    print_model_summary,
    print_separator,
)


def demo_basic_formatting():
    """Demo 1: Basic formatting functions."""
    print_header("Logging & Progress Demo")
    print_section("Basic Formatting")
    
    print_info("This is an informational message")
    print_success("Operation completed successfully!")
    print_warning("This is a warning message")
    print_error("This is an error message")
    
    print_separator()


def demo_tables():
    """Demo 2: Tables."""
    print_section("Tables")
    
    # Metrics table
    metrics = {
        'Loss (MSE)': 0.2451,
        'MAE': 0.1823,
        'RMSE': 0.4950,
        'Samples/sec': 1234,
    }
    print_metric_table(metrics, title="Training Metrics")
    
    print()
    
    # Configuration table
    config = {
        'model': 'ChessCNN',
        'batch_size': 256,
        'learning_rate': 0.001,
        'epochs': 50,
        'optimizer': 'Adam',
    }
    print_config(config, title="Experiment Configuration")
    
    print()
    
    # Comparison table
    comparison = {
        'Simple (13ch)': {'loss': 0.245, 'mae': 0.182, 'params': 2.3e6},
        'Essential (19ch)': {'loss': 0.231, 'mae': 0.174, 'params': 2.4e6},
        'Full (23ch)': {'loss': 0.219, 'mae': 0.165, 'params': 2.4e6},
    }
    print_comparison_table(comparison, title="Model Comparison")
    
    print()


def demo_model_summary():
    """Demo 3: Model summary."""
    print_section("Model Summary")
    
    print_model_summary(
        model_name="ChessCNN",
        parameters=2376961,
        input_shape=(23, 8, 8),
        output_shape=(1,),
        additional_info={
            'Base Channels': 64,
            'Conv Blocks': 4,
            'Dropout (conv)': 0.3,
            'Dropout (fc)': 0.5,
        }
    )
    
    print()


def demo_simple_progress():
    """Demo 4: Simple progress bar."""
    print_section("Simple Progress Bar")
    
    with SimpleProgress("Processing data", total=100) as progress:
        for i in range(100):
            time.sleep(0.02)
            progress.update(1)
    
    print_success("Processing complete!")
    print()


def demo_data_load_progress():
    """Demo 5: Data loading progress."""
    print_section("Data Loading Progress")
    
    progress = DataLoadProgress()
    progress.start("Loading chess positions", total=1000)
    
    for i in range(1000):
        time.sleep(0.001)
        progress.update(1)
    
    progress.finish()
    print_success("Data loaded successfully!")
    print()


def demo_training_progress():
    """Demo 6: Training progress with nested bars."""
    print_section("Training Progress (Nested Bars)")
    
    num_epochs = 5
    batches_per_epoch = 20
    
    with TrainingProgress() as progress:
        # Create epoch bar
        progress.create_epoch_bar(num_epochs, "Training")
        
        for epoch in range(num_epochs):
            # Create/reset batch bar
            progress.create_batch_bar(batches_per_epoch, f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch in range(batches_per_epoch):
                # Simulate training
                time.sleep(0.05)
                
                # Update metrics
                loss = 1.0 / (epoch + 1) - (batch / batches_per_epoch) * 0.2
                mae = loss * 0.7
                progress.update_metrics({
                    'loss': loss,
                    'mae': mae,
                    'lr': 0.001,
                })
                
                # Update batch progress
                progress.update_batch(1)
            
            # Update epoch progress
            progress.update_epoch(1)
    
    print_success("Training complete!")
    print()


def demo_experiment_logger():
    """Demo 7: Experiment logger."""
    print_section("Experiment Logger")
    
    # Create logger
    logger = ExperimentLogger(name="demo")
    logger.setup_experiment("logging_demo_experiment")
    
    logger.header("Starting Experiment")
    
    logger.info("Loading model...")
    logger.success("Model loaded successfully")
    
    logger.info("Loading data...")
    logger.success("Data loaded: 100,000 samples")
    
    logger.section("Training Configuration")
    logger.config({
        'model': {
            'type': 'CNN',
            'base_channels': 64,
            'dropout': 0.3,
        },
        'training': {
            'batch_size': 256,
            'learning_rate': 0.001,
            'epochs': 50,
        }
    })
    
    logger.section("Training Metrics")
    for epoch in range(3):
        logger.metric("epoch", epoch + 1)
        logger.metrics({
            'train_loss': 0.5 - epoch * 0.1,
            'val_loss': 0.55 - epoch * 0.09,
            'train_mae': 0.35 - epoch * 0.05,
            'val_mae': 0.38 - epoch * 0.045,
        }, step=epoch + 1)
    
    logger.success("Experiment completed successfully!")
    logger.info(f"Log file saved to: {logger.log_file}")
    
    logger.close()
    print()


def demo_combined():
    """Demo 8: Combined logger + progress."""
    print_section("Combined Logging & Progress")
    
    logger = ExperimentLogger(name="combined_demo")
    logger.header("Training with Live Logging")
    
    logger.info("Initializing training...")
    
    num_epochs = 3
    batches_per_epoch = 15
    
    with TrainingProgress() as progress:
        progress.create_epoch_bar(num_epochs, "Training")
        
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            progress.create_batch_bar(batches_per_epoch, f"Epoch {epoch + 1}")
            
            epoch_loss = 0.0
            for batch in range(batches_per_epoch):
                time.sleep(0.05)
                
                batch_loss = 1.0 / (epoch + 1) - (batch / batches_per_epoch) * 0.2
                epoch_loss += batch_loss
                
                progress.update_metrics({'loss': batch_loss})
                progress.update_batch(1)
            
            avg_loss = epoch_loss / batches_per_epoch
            logger.info(f"Epoch {epoch + 1} complete - Avg Loss: {avg_loss:.4f}")
            
            progress.update_epoch(1)
    
    logger.success("Training complete!")
    
    # Final results
    final_metrics = {
        'Final Loss': 0.234,
        'Final MAE': 0.167,
        'Training Time': '5m 23s',
        'Samples/sec': 1234,
    }
    logger.table(final_metrics, title="Final Results")
    
    print()


def main():
    """Run all demos."""
    print_header("Chess AI - Logging & Progress System Demo")
    
    print_info("This demo showcases all logging and progress features")
    print_separator()
    print()
    
    demo_basic_formatting()
    demo_tables()
    demo_model_summary()
    demo_simple_progress()
    demo_data_load_progress()
    demo_training_progress()
    demo_experiment_logger()
    demo_combined()
    
    print_header("Demo Complete!")
    print_success("All features demonstrated successfully")
    print_info("Check experiments/logs/ for experiment log files")


if __name__ == '__main__':
    main()

