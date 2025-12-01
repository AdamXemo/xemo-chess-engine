# Chess Position Evaluator

Training neural networks to evaluate chess positions using PyTorch. The goal is to build models that can accurately assess positions and eventually play decent chess.

## What's This About?

I'm training neural networks to predict position evaluations using a dataset of 2.2M+ positions from Lichess 2017 (not full dataset) with Stockfish analysis. This is not my 1st attempt at this project - previous versions were either too simple / poorly structured or overly complicated and mostly AI-generated, so this time I'm focusing on understanding every piece of code.

**Current Phase**: Position evaluation (board → evaluation score)

**Future Plans**: 
- Move prediction network
- Combined policy-value network for MCTS

## Dataset

The dataset contains FEN positions paired with Stockfish evaluations:
- 2,244,390 positions from Lichess 2017 database
- Evaluations in range [-100, 100] where ±100 = mate
- Positive = White advantage, Negative = Black advantage

Format: `FEN,Evaluation`

**Note**: Dataset not included in repo (135MB). You'll need to provide your own chess position dataset in `data/chess_data.csv` with the format above.

## Input Representations

Three different bitboard representations to experiment with:

| Type | Channels | What's Included |
|------|----------|-----------------|
| Simple | 13 | Piece positions + turn |
| Essential | 19 | Pieces + turn + castling + en passant + check |
| Full | 23 | Essential + attack maps + mobility maps |

All representations use binary channels (0 or 1) with shape `(C, 8, 8)`.

## Project Structure

```
chess_ai/
├── data/               # Dataset
├── src/
│   ├── data/          # Data loading & bitboard conversion
│   ├── models/        # Neural network architectures (TODO)
│   ├── training/      # Training loops (TODO)
│   ├── evaluation/    # Model evaluation (TODO)
│   ├── config/        # Configuration management
│   └── utils/         # Utilities (TODO)
├── experiments/       # Logs, checkpoints, results
├── tests/            # Test suite
├── examples/         # Usage examples
└── docs/             # Detailed documentation
```

## Quick Start

```python
from src.data import ChessDataLoader
from torch.utils.data import DataLoader

# Load and split data
train_ds, val_ds, test_ds = ChessDataLoader.create_datasets(
    csv_path='data/chess_data.csv',
    representation='full',
    max_samples=100000  # None for full dataset
)

# Create DataLoader
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

# Iterate
for bitboards, evaluations in train_loader:
    # bitboards: (batch_size, 23, 8, 8)
    # evaluations: (batch_size, 1)
    pass
```

## Configuration

Experiments use Python dataclasses for configuration:

```python
from src.config import ExperimentConfig, DataConfig, TrainingConfig

config = ExperimentConfig(
    experiment_name="baseline_cnn",
    data=DataConfig(representation='full'),
    training=TrainingConfig(batch_size=256, learning_rate=0.001)
)

config.save('experiments/configs/baseline_cnn.json')
```

## Running Tests

```bash
# Test data loading pipeline
python tests/test_data_loading.py

# See usage examples
python examples/example_usage.py
```

## What's Implemented

- [x] Data loading from CSV
- [x] Three bitboard representations
- [x] PyTorch Dataset & DataLoader integration
- [x] Configuration system
- [ ] Model architectures (CNN, ResNet)
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Model comparison experiments

## Hardware

- Training on NVIDIA RTX 4060 (16GB RAM)
- Quick experiments: ~15 minutes
- Full training runs: up to 10 hours
- May scale to Google Colab later

## Dependencies

```bash
pip install -r requirements.txt
```

Core: PyTorch 2.0+, python-chess, NumPy

## Documentation

See `docs/PROJECT_SCOPE.md` for detailed specifications and design decisions.

## License

TBD
