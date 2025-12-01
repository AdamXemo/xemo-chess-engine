# Logging and Progress System Documentation

## Overview

Professional logging and progress bar system using the `rich` library for beautiful terminal output and Python's `logging` module for structured file-based experiment logs.

## Features

- Beautiful terminal output with colors, boxes, and tables
- Nested progress bars for training (epochs + batches)
- Experiment-specific log files with timestamps
- Live metric updates during training
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Consistent formatting across all project components

## Components

### 1. Formatting Utilities (`src/utils/formatting.py`)

Provides functions for consistent terminal output formatting.

#### Basic Print Functions

```python
from src.utils import print_header, print_section, print_success

print_header("My Application")
print_section("Testing Phase")
print_success("Test passed!")
print_error("Test failed!")
print_warning("Warning message")
print_info("Info message")
```

#### Tables

```python
from src.utils import print_metric_table

metrics = {
    'Loss': 0.245,
    'MAE': 0.182,
    'RMSE': 0.495
}
print_metric_table(metrics, title="Training Metrics")
```

#### Model Summary

```python
from src.utils import print_model_summary

print_model_summary(
    model_name="ChessCNN",
    parameters=2376961,
    input_shape=(23, 8, 8),
    output_shape=(1,),
    additional_info={'Base Channels': 64}
)
```

#### Comparison Tables

```python
from src.utils import print_comparison_table

comparison = {
    'Model A': {'loss': 0.24, 'mae': 0.18},
    'Model B': {'loss': 0.22, 'mae': 0.16}
}
print_comparison_table(comparison, title="Model Comparison")
```

### 2. Experiment Logger (`src/utils/logging.py`)

Unified logger combining terminal output and file logging.

#### Basic Usage

```python
from src.utils import ExperimentLogger

# Create logger
logger = ExperimentLogger(name="my_experiment")

# Setup experiment file logging
logger.setup_experiment("my_experiment_name")

# Log messages
logger.info("Training started")
logger.success("Epoch completed")
logger.warning("Learning rate adjusted")
logger.error("Training failed")

# Log with structure
logger.header("Training Phase")
logger.section("Epoch 1")

# Log metrics
logger.metric("loss", 0.245, step=1)
logger.metrics({'loss': 0.245, 'mae': 0.182}, step=1)

# Log tables
logger.table({'Loss': 0.245, 'MAE': 0.182})

# Log configuration
logger.config({
    'model': 'CNN',
    'batch_size': 256
})

# Close when done
logger.close()
```

#### Global Logger

```python
from src.utils import get_logger, setup_experiment

# Get global logger instance
logger = get_logger()

# Or setup experiment directly
logger = setup_experiment("experiment_name")
```

### 3. Progress Bars (`src/utils/progress.py`)

Professional progress bars for training and data loading.

#### Training Progress (Nested Bars)

```python
from src.utils import TrainingProgress

num_epochs = 50
batches_per_epoch = 100

with TrainingProgress() as progress:
    # Create epoch bar
    progress.create_epoch_bar(num_epochs, "Training")
    
    for epoch in range(num_epochs):
        # Create batch bar for this epoch
        progress.create_batch_bar(batches_per_epoch, f"Epoch {epoch+1}")
        
        for batch in range(batches_per_epoch):
            # Your training code here
            loss = train_batch()
            
            # Update metrics (shown live)
            progress.update_metrics({
                'loss': loss,
                'mae': compute_mae(),
            })
            
            # Update progress
            progress.update_batch(1)
        
        # Update epoch progress
        progress.update_epoch(1)
```

#### Data Loading Progress

```python
from src.utils import DataLoadProgress

progress = DataLoadProgress()
progress.start("Loading positions", total=10000)

for i in range(10000):
    # Load data
    progress.update(1)

progress.finish()
```

#### Simple Progress

```python
from src.utils import SimpleProgress

with SimpleProgress("Processing", total=100) as progress:
    for i in range(100):
        # Do work
        progress.update(1)
```

## Visual Examples

### Headers and Sections

```
╭──────────────────────────────────────────────────────╮
│           Chess Neural Network Training              │
╰──────────────────────────────────────────────────────╯

─────────────────── Training Phase 1 ───────────────────
```

### Success/Error Messages

```
✓ Model loaded successfully
✗ Training failed
⚠ Learning rate adjusted
ℹ Starting epoch 5
```

### Metrics Table

```
        Training Metrics        
╭─────────────┬────────╮
│ Metric      │  Value │
├─────────────┼────────┤
│ Loss (MSE)  │ 0.2451 │
│ MAE         │ 0.1823 │
│ RMSE        │ 0.4950 │
│ Samples/sec │  1,234 │
╰─────────────┴────────╯
```

### Model Summary

```
      Model: ChessCNN       
╔══════════════════╦═══════════╗
║ Total Parameters ║ 2,376,961 ║
║ Input Shape      ║ (23,8,8)  ║
║ Output Shape     ║ (1,)      ║
╚══════════════════╩═══════════╝
```

### Progress Bars

```
Training ━━━━━━━╸━━━━━━━━━━━━━ 15% 8/50  0:12:34 • 1:23:45
  Batches (loss: 0.245 | mae: 0.182) ━━━━━━━━━━ 65% 65/100 0:00:12 • 0:00:06
```

## Integration Examples

### Test Files

Both test files have been updated to use the new logging system:

```python
from src.utils import (
    print_header,
    print_section,
    print_success,
    print_metric_table,
)

print_header("Test Suite")
print_section("Testing Module")

# Run tests
test_result = run_test()

if test_result:
    print_success("Test passed!")
else:
    print_error("Test failed!")

# Show results
print_metric_table(results, title="Test Results")
```

### Training Loop (Future)

```python
from src.utils import ExperimentLogger, TrainingProgress

# Setup logging
logger = ExperimentLogger()
logger.setup_experiment("cnn_training")

logger.header("Training ChessCNN")
logger.config(config)

# Training with progress
with TrainingProgress() as progress:
    progress.create_epoch_bar(num_epochs)
    
    for epoch in range(num_epochs):
        progress.create_batch_bar(num_batches)
        
        for batch in train_loader:
            loss = train_step(batch)
            progress.update_metrics({'loss': loss})
            progress.update_batch(1)
        
        # Log epoch results
        logger.metrics({
            'train_loss': train_loss,
            'val_loss': val_loss
        }, step=epoch)
        
        progress.update_epoch(1)

logger.success("Training complete!")
logger.close()
```

## File Logging

When you setup an experiment, logs are saved to:
```
experiments/logs/experiment_name_20251201_123456.log
```

Format:
```
2025-12-01 12:34:56 | INFO     | Training started
2025-12-01 12:35:01 | INFO     | Epoch 1 complete
2025-12-01 12:35:01 | INFO     | train_loss: 0.245
```

## Best Practices

1. **Use appropriate log levels**:
   - `debug()` for detailed debugging info
   - `info()` for general information
   - `warning()` for warnings
   - `error()` for errors

2. **Use structured logging**:
   ```python
   # Good
   logger.metrics({'loss': 0.245, 'mae': 0.182}, step=epoch)
   
   # Less good
   logger.info(f"Loss: {loss}, MAE: {mae}")
   ```

3. **Use tables for multiple values**:
   ```python
   # Good
   print_metric_table(metrics)
   
   # Less good
   for k, v in metrics.items():
       print(f"{k}: {v}")
   ```

4. **Use progress bars for long operations**:
   - Data loading
   - Training loops
   - Batch processing
   - Any operation > 1 second

5. **Close loggers when done**:
   ```python
   logger.close()  # Flushes file buffers
   ```

## Color Scheme

Consistent colors across all output:

- **Cyan**: Headers, sections, info
- **Green**: Success messages
- **Yellow**: Warnings
- **Red**: Errors
- **Magenta**: Values, highlights
- **Dim**: Separators, less important text

## Dependencies

- `rich>=13.0.0` - Terminal formatting and progress bars
- Python `logging` - Standard library logging

## Examples

Run the demo to see all features:
```bash
python examples/logging_demo.py
```

Run updated tests to see integration:
```bash
python tests/test_data_loading.py
python tests/test_model.py
```

