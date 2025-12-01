# Project Scope

## Overview

Training neural networks to evaluate chess positions using the Lichess 2017 database with Stockfish analysis. Focus is on clean, understandable code and systematic comparison of different architectures and input representations.

### Project Goals

The project consists of three phases, with Phase 1 being the current focus:

1. **Phase 1**: Position evaluation network (board → evaluation score)
2. **Phase 2**: Move prediction network (board → move probabilities)
3. **Phase 3**: Combined policy-value network for MCTS integration

---

## Data Specifications

### Dataset Information

- **Source**: Lichess 2017 database with Stockfish evaluations
- **Size**: 2,244,390+ position samples
- **Format**: CSV file with two columns separated by comma
- **Location**: `data/chess_data.csv`

### Data Format

```
FEN,Evaluation
```

**Example entries**:
```
2kr4/pp1npp2/2p1b1rb/q3B1pQ/P1pP3p/2P1PN1P/4BPP1/1R3RK1 w - - 18,-0.63
rnbqk1nr/p1ppppbp/1p4p1/8/2PPP3/2N5/PP3PPP/R1BQKBNR b KQkq - 4,0.65
```

### Evaluation Score Properties

- **Range**: -100.0 to +100.0
- **Interpretation**: 
  - Positive values indicate White advantage
  - Negative values indicate Black advantage
  - Always from White's perspective (not side-to-move)
- **Special values**: ±100.0 represents forced mate
- **Units**: Approximately equivalent to centipawns divided by 100

---

## Input Representations

The project will implement and compare three distinct bitboard representations of chess positions. All representations use binary channels (0 or 1) with shape `(C, 8, 8)` where `C` is the number of channels.

### Representation 1: Simple (13 channels)

*Minimal representation with only piece positions and turn indicator.*

**Channel allocation**:
- Channels 0-11: Piece positions (one channel per piece type)
  - 0: White Pawns
  - 1: White Knights
  - 2: White Bishops
  - 3: White Rooks
  - 4: White Queens
  - 5: White King
  - 6: Black Pawns
  - 7: Black Knights
  - 8: Black Bishops
  - 9: Black Rooks
  - 10: Black Queens
  - 11: Black King
- Channel 12: Turn indicator (1 = White to move, 0 = Black to move)

### Representation 2: Essential (N channels)

*Core chess features without derived features like attack/mobility maps.*

**Channel allocation**:
- Channels 0-11: Piece positions (same as Simple)
- Channel 12: Turn indicator
- Channel 13: En passant target square
- Channels 14-17: Castling rights (4 separate channels)
  - 14: White kingside castling
  - 15: White queenside castling
  - 16: Black kingside castling
  - 17: Black queenside castling
- Channel 18: Check indicator

**Total**: 19 channels

### Representation 3: Full (23 channels)

*Complete representation including computed tactical features.*

**Channel allocation**:
- Channels 0-11: Piece positions
- Channel 12: Turn indicator
- Channel 13: En passant target square
- Channels 14-17: Castling rights
- Channels 18-19: Attack maps
  - 18: White attacks (squares attacked by White pieces)
  - 19: Black attacks (squares attacked by Black pieces)
- Channels 20-21: Mobility maps
  - 20: White mobility (squares from which White can move)
  - 21: Black mobility (squares from which Black can move)
- Channel 22: Check indicator

**Total**: 23 channels

> **Note**: All plane-wide indicators (turn, castling rights, check) fill the entire 8×8 plane with 0 or 1.

---

## Neural Network Architectures

### Architecture 1: Convolutional Neural Network (CNN)

A baseline convolutional architecture for initial experiments.

**Key characteristics**:
- Multiple convolutional layers with ReLU activation
- Batch normalization for training stability
- Global pooling or flattening before fully connected layers
- Single output neuron with linear activation (regression)

### Architecture 2: Residual Network (ResNet)

Primary architecture inspired by AlphaZero's approach.

**Key characteristics**:
- Initial convolutional block
- Multiple residual blocks (skip connections)
- Each residual block contains:
  - Two convolutional layers
  - Batch normalization
  - ReLU activation
  - Skip connection
- Value head for position evaluation
- Scalable depth (configurable number of residual blocks)

### Architecture 3+: Experimental Architectures

Future architectures to be explored:

- **Attention mechanisms**: Self-attention layers for capturing long-range piece interactions
- **Transformer-based**: Pure transformer or hybrid CNN-Transformer architectures
- **Squeeze-and-Excitation networks**: Channel-wise attention mechanisms
- **Hybrid architectures**: Combining multiple approaches

---

## Training Pipeline

### Data Processing

1. **Loading**: Read CSV file into memory (2M+ samples fit comfortably in 16GB RAM)
2. **Parsing**: Extract FEN string and evaluation score
3. **Conversion**: Convert FEN to selected bitboard representation using `BitboardConverter`
4. **Normalization**: Evaluation scores may be normalized or kept in [-100, +100] range
5. **Splitting**: Train/validation/test split (suggested: 80/10/10)

### Training Configuration

**Hardware specifications**:
- GPU: NVIDIA RTX 4060
- RAM: 16 GB
- Training time constraints:
  - Quick tests: ~15 minutes
  - Experimental runs: ~1 hour
  - Full training: ~10 hours

**Training parameters** (to be determined through experimentation):
- Batch size: TBD (constrained by GPU memory)
- Learning rate: TBD (with learning rate scheduling)
- Optimizer: Adam or AdamW (to be compared)
- Loss function: Mean Squared Error (MSE) or Huber Loss
- Number of epochs: TBD based on convergence
- Regularization: Dropout, weight decay

### Checkpointing Strategy

- Save model checkpoint after every epoch
- Track and save best model based on validation loss
- Store training metrics for analysis
- Include optimizer state for resumable training

---

## Evaluation Methodology

### Quantitative Metrics

1. **Loss metrics**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)

2. **Position-specific evaluation**:
   - Error distribution analysis
   - Performance on different position types (opening, middlegame, endgame)
   - Accuracy on tactical vs positional evaluations

### Qualitative Evaluation

1. **Known positions testing**:
   - Test on famous positions with known evaluations
   - Analyze predictions on tactical puzzles
   - Compare to Stockfish evaluations at various depths

2. **Engine play**:
   - Integrate with chess engine using evaluation function
   - Play against Stockfish at different strength levels
   - Play against other trained models
   - Measure Elo rating approximation

### Comparative Analysis

- Compare performance across different bitboard representations
- Compare different neural network architectures
- Analyze computational efficiency (inference time, model size)
- Document trade-offs between model complexity and performance

---

## Project Structure

```
chess_ai/
├── data/
│   ├── chess_data.csv           # Raw dataset
│   └── processed/               # Preprocessed data (if needed)
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bitboard.py          # Bitboard conversion utilities
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   └── preprocessing.py     # Data preprocessing utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base model class
│   │   ├── cnn.py               # CNN architecture
│   │   ├── resnet.py            # ResNet architecture
│   │   └── experimental/        # Future experimental models
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop and logic
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── callbacks.py         # Training callbacks (logging, checkpointing)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py         # Model evaluation utilities
│   │   ├── positions.py         # Test position definitions
│   │   └── play.py              # Engine play integration
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base_config.py       # Base configuration class
│   │   ├── model_configs.py     # Model-specific configurations
│   │   └── training_configs.py  # Training hyperparameters
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # Logging utilities
│       └── visualization.py     # Plotting and visualization
│
├── experiments/
│   ├── logs/                    # Training logs
│   ├── checkpoints/             # Model checkpoints
│   └── results/                 # Experiment results and analysis
│
├── tests/
│   ├── test_bitboard.py
│   ├── test_dataset.py
│   └── test_models.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration
│
├── requirements.txt             # Python dependencies
├── PROJECT_SCOPE.md            # This document
└── README.md                   # Project readme
```

---

## Development Phases

### Phase 1.1: Foundation (Current)

- [x] Define project scope and specifications
- [ ] Implement bitboard conversion utilities
- [ ] Implement dataset loading and PyTorch Dataset class
- [ ] Implement configuration management system
- [ ] Set up basic project structure

### Phase 1.2: Model Implementation

- [ ] Implement abstract base model class
- [ ] Implement CNN architecture
- [ ] Implement ResNet architecture
- [ ] Add model unit tests

### Phase 1.3: Training Infrastructure

- [ ] Implement training loop
- [ ] Implement metrics tracking
- [ ] Implement checkpointing system
- [ ] Set up experiment logging

### Phase 1.4: Evaluation and Analysis

- [ ] Implement evaluation metrics
- [ ] Create test position suite
- [ ] Train baseline models
- [ ] Compare architectures and representations
- [ ] Document findings

### Phase 1.5: Refinement

- [ ] Optimize hyperparameters
- [ ] Improve model architecture based on results
- [ ] Conduct ablation studies
- [ ] Final training run on best configuration

---

## Future Work (Phases 2 & 3)

### Phase 2: Move Prediction Network

- Expand dataset to include move information
- Implement policy head for move prediction
- Train move prediction models
- Evaluate move prediction accuracy

### Phase 3: Combined Policy-Value Network

- Implement combined architecture with policy and value heads
- Integrate with Monte Carlo Tree Search (MCTS)
- Train combined model
- Implement full chess engine with MCTS
- Benchmark against established engines

---

## Technical Dependencies

### Core Dependencies

- **PyTorch**: Deep learning framework (version ≥ 2.0)
- **python-chess**: Chess logic and FEN parsing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation (if needed)

### Development Dependencies

- **pytest**: Unit testing
- **tensorboard**: Training visualization (optional)
- **matplotlib/seaborn**: Result plotting
- **tqdm**: Progress bars

### Hardware Requirements

- **Minimum**: 8GB RAM, any GPU with CUDA support
- **Recommended**: 16GB RAM, NVIDIA RTX 4060 or better
- **Future**: Google Colab GPU or dedicated training hardware

---

## Experiment Tracking

### Configuration Management

Each experiment will be defined by a Python configuration file specifying:

- Model architecture and hyperparameters
- Input representation (13/19/23 channels)
- Training hyperparameters
- Data split configuration
- Random seeds for reproducibility

### Logging and Versioning

- Each experiment gets unique identifier (timestamp-based or manual naming)
- All configurations saved alongside checkpoints
- Training metrics logged at each epoch
- Git commit hash recorded for code version tracking

### Result Documentation

- Quantitative results stored in structured format
- Comparative visualizations generated automatically
- Markdown reports for major experiments
- Best models archived with full metadata

---

## Success Criteria

### Phase 1 Success Metrics

1. **Training stability**: Models converge without divergence or collapse
2. **Evaluation accuracy**: MAE < 0.5 (or equivalent metric) on test set
3. **Known positions**: Reasonable evaluations on standard test positions
4. **Comparative insights**: Clear understanding of which architectures and representations perform best
5. **Code quality**: Clean, documented, understandable codebase

### Long-term Success Criteria

- Competitive play against intermediate chess engines
- Fast inference time (< 10ms per position)
- Extensible codebase supporting Phases 2 and 3
- Publishable experimental findings

---

## Risk Mitigation

### Technical Risks

- **Overfitting**: Use validation set, regularization, early stopping
- **Dataset bias**: Analyze error patterns, consider data augmentation (board flipping)
- **Training instability**: Careful learning rate tuning, gradient clipping, batch normalization
- **Memory constraints**: Batch size optimization, gradient accumulation if needed

### Project Risks

- **Scope creep**: Focus on Phase 1, resist adding features prematurely
- **Time management**: Set realistic training time expectations
- **Hardware limitations**: Plan for cloud GPU usage if local hardware insufficient

---

## Notes and Considerations

### Data Augmentation

Board horizontal flipping can double effective dataset size:
- Flip board left-right
- Flip file letters in FEN notation
- Preserve evaluation (White perspective unchanged)

### Mate Score Handling

Current approach treats mate as ±100. Alternative approaches:

- Clip extreme values to ±10 or ±20
- Separate classification head for mate detection
- Weighted loss function for mate positions

### Evaluation Perspective

All evaluations are from White's perspective. Model must learn this convention or be explicitly designed to handle it (e.g., input channel for side-to-move).

### Computational Efficiency

For real-time play, inference speed matters:
- Profile model inference time
- Consider model quantization or pruning
- Batch inference for position analysis

---

*Document Version: 1.0*  
*Last Updated: 2025-12-01*  
*Status: Phase 1.1 - Foundation*

