# Training System Documentation

## Overview

Complete training pipeline for chess position evaluation models with metrics tracking, checkpointing, early stopping, and beautiful progress visualization.

## Components

### 1. Trainer Class (`src/training/trainer.py`)

Main orchestrator for the training process.

**Key Features:**
- Automatic device selection (CPU/CUDA)
- Training and validation loops
- Checkpoint management (every epoch + best model)
- Learning rate scheduling
- Gradient clipping
- Early stopping
- Full integration with logging system

**Basic Usage:**

```python
from src.models import ChessCNN
from src.data import ChessDataLoader
from src.training import Trainer
from src.config import ExperimentConfig
from torch.utils.data import DataLoader

# Load data
train_ds, val_ds, test_ds = ChessDataLoader.create_datasets(
    'data/chess_data.csv',
    representation='full',
    max_samples=100000
)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

# Create model
model = ChessCNN(input_channels=23)

# Setup config
config = ExperimentConfig(
    experiment_name="my_experiment",
    # ... configuration ...
)

# Train
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()

# Evaluate
test_loader = DataLoader(test_ds, batch_size=256)
test_metrics = trainer.evaluate(test_loader)
```

### 2. Metrics Tracking (`src/training/metrics.py`)

Track and compute metrics during training.

**MetricsTracker Class:**
```python
from src.training import MetricsTracker

tracker = MetricsTracker(metrics_list=['mse', 'mae', 'rmse'])

# During training loop
for batch in train_loader:
    outputs = model(inputs)
    tracker.update(outputs, targets)

# After epoch
metrics = tracker.compute()
tracker.add_to_history(metrics, epoch=0, phase='train')

# Save history
tracker.save_history('training_history.json')
```

**Available Metrics:**
- `mse` - Mean Squared Error
- `mae` - Mean Absolute Error  
- `rmse` - Root Mean Squared Error

### 3. Early Stopping (`src/training/early_stopping.py`)

Stop training when validation loss plateaus.

```python
from src.training import EarlyStopping

early_stop = EarlyStopping(
    patience=10,        # Wait 10 epochs
    min_delta=0.0001,   # Minimum improvement
    mode='min'          # For loss (use 'max' for accuracy)
)

for epoch in range(num_epochs):
    val_loss = validate()
    
    if early_stop(val_loss):
        print("Early stopping triggered!")
        break
    
    if early_stop.improved:
        print("Model improved!")
```

### 4. Training Utilities (`src/training/utils.py`)

Helper functions for training.

```python
from src.training import (
    set_seed,
    get_lr,
    save_training_curves,
    format_time
)

# Set random seed for reproducibility
set_seed(42)

# Get current learning rate
lr = get_lr(optimizer)

# Save training curves
save_training_curves(
    history=trainer.metrics_tracker.get_history(),
    save_path='training_curves.png',
    metrics=['mse', 'mae']
)

# Format time nicely
print(format_time(elapsed_seconds))  # "1h 23m"
```

## Configuration

Training configuration in `ExperimentConfig`:

```python
from src.config import ExperimentConfig, TrainingConfig

config = ExperimentConfig(
    experiment_name="cnn_baseline",
    training=TrainingConfig(
        # Optimization
        batch_size=256,
        num_epochs=50,
        learning_rate=0.001,
        weight_decay=0.0001,
        optimizer='adam',  # 'adam', 'adamw', 'sgd'
        
        # Learning rate scheduling
        use_scheduler=True,
        scheduler_type='reduce_on_plateau',
        scheduler_patience=5,
        scheduler_factor=0.5,
        
        # Loss function
        loss_function='mse',  # 'mse', 'mae', 'huber'
        
        # Regularization
        gradient_clip=1.0,
        
        # Checkpointing
        save_every_epoch=True,
        checkpoint_dir='experiments/checkpoints',
        
        # Early stopping
        use_early_stopping=True,
        early_stopping_patience=10,
        
        # Validation
        validate_every=1  # Validate every N epochs
    )
)
```

## Training Loop Flow

```
1. Setup
   ├─ Model to device
   ├─ Initialize optimizer
   ├─ Initialize scheduler
   ├─ Setup loss function
   ├─ Setup logger
   └─ Setup metrics tracker

2. For each epoch:
   ├─ Training Phase
   │  ├─ Model in train mode
   │  ├─ For each batch:
   │  │  ├─ Forward pass
   │  │  ├─ Compute loss
   │  │  ├─ Backward pass
   │  │  ├─ Gradient clipping (if enabled)
   │  │  ├─ Optimizer step
   │  │  └─ Update metrics
   │  └─ Compute epoch metrics
   │
   ├─ Validation Phase
   │  ├─ Model in eval mode
   │  ├─ No gradients
   │  ├─ For each batch:
   │  │  ├─ Forward pass
   │  │  └─ Update metrics
   │  └─ Compute epoch metrics
   │
   ├─ Logging
   │  └─ Log train/val metrics
   │
   ├─ Learning Rate Update
   │  └─ Scheduler step
   │
   ├─ Checkpointing
   │  ├─ Save regular checkpoint
   │  └─ Save best model if improved
   │
   └─ Early Stopping Check
      └─ Stop if no improvement

3. Finalization
   ├─ Log training summary
   ├─ Save training history
   ├─ Save training curves plot
   └─ Close logger
```

## Checkpoint Structure

Checkpoints are saved as `.pth` files:

```python
{
    'epoch': 10,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},
    'best_val_loss': 0.234,
    'metrics_history': {...},
    'config': {...},
    'early_stopping_state': {...}
}
```

**Checkpoint Files:**
- `experiments/checkpoints/{experiment_name}/epoch_{N}.pth` - Every epoch
- `experiments/checkpoints/{experiment_name}/best_model.pth` - Best model
- `experiments/checkpoints/{experiment_name}/last_checkpoint.pth` - Most recent

## Results and Logs

**Training History:**
- `experiments/results/{experiment_name}/training_history.json`

**Training Curves:**
- `experiments/results/{experiment_name}/training_curves.png`

**Experiment Logs:**
- `experiments/logs/{experiment_name}_{timestamp}.log`

## Resuming Training

```python
# Create trainer
trainer = Trainer(model, train_loader, val_loader, config)

# Load checkpoint
trainer.load_checkpoint('experiments/checkpoints/my_exp/best_model.pth')

# Resume training
trainer.train()
```

## Examples

### Quick Training Demo

```bash
python examples/training_demo.py
```

Trains a small model on 10k samples for 10 epochs (takes ~2 minutes).

### Full Training Example

```python
from src.models import ChessCNN
from src.data import ChessDataLoader
from src.training import Trainer
from src.config import ExperimentConfig, DataConfig, TrainingConfig, CNNConfig
from torch.utils.data import DataLoader

# Configuration
config = ExperimentConfig(
    experiment_name="cnn_full_training",
    data=DataConfig(
        representation='full',
        max_samples=None,  # Use all data
    ),
    model=CNNConfig(
        base_channels=64,
        dropout_conv=0.3,
        dropout_fc=0.5
    ),
    training=TrainingConfig(
        batch_size=256,
        num_epochs=50,
        learning_rate=0.001,
        use_early_stopping=True,
        early_stopping_patience=10
    )
)

# Data
train_ds, val_ds, test_ds = ChessDataLoader.create_datasets(
    config.data.csv_path,
    representation=config.data.representation
)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=256, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=256, num_workers=4)

# Model
model = ChessCNN(input_channels=23, base_channels=64)

# Train
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()

# Evaluate
test_metrics = trainer.evaluate(test_loader)
print(f"Test MSE: {test_metrics['mse']:.4f}")
print(f"Test MAE: {test_metrics['mae']:.4f}")
```

## Progress Visualization

During training, you'll see:

```
╭─────────────────────────────────────────╮
│          Training Started               │
╰─────────────────────────────────────────╯

Training ━━━━━━━━━━━━━━━━━━━━ 15% 8/50  0:12:34 • 1:23:45
  Batches (loss: 0.245 | lr: 0.001) ━━━━ 65% 200/312 0:00:12 • 0:00:06

          Epoch 8          
╭────────────┬──────────╮
│ Metric     │    Value │
├────────────┼──────────┤
│ Train Loss │  0.2451  │
│ Val Loss   │  0.2398  │
│ Train MAE  │  0.1823  │
│ Val MAE    │  0.1801  │
│ LR         │ 1.00e-03 │
│ Time       │     3m   │
╰────────────┴──────────╯
```

## Testing

Run training tests:
```bash
python tests/test_training.py
```

Tests cover:
- Metrics computation
- MetricsTracker functionality
- Early stopping
- Training utilities
- Single epoch training
- Checkpoint save/load

## Tips

1. **Start Small**: Use `max_samples` for quick experiments
2. **Monitor Validation**: Watch val loss to avoid overfitting
3. **Learning Rate**: Start with 1e-3, adjust if needed
4. **Early Stopping**: Use patience=10 as a safe default
5. **Checkpoints**: Keep best model, delete intermediate ones to save space
6. **Batch Size**: Larger = faster but needs more GPU memory
7. **Gradient Clipping**: Use 1.0 if training unstable

## Troubleshooting

**Training is slow:**
- Increase batch size
- Use more `num_workers` in DataLoader
- Ensure model is on GPU (`device='cuda'`)

**Loss not decreasing:**
- Check learning rate (try lower)
- Check data normalization
- Verify model architecture
- Check for bugs in data loading

**Out of memory:**
- Reduce batch size
- Use smaller model (lower `base_channels`)
- Use gradient accumulation

**Early stopping too early:**
- Increase patience
- Reduce min_delta
- Check if validation set is too small

