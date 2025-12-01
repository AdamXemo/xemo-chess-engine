# CNN Architecture Documentation

## Implemented: Progressive Depth CNN

A convolutional neural network that gradually increases feature depth while maintaining spatial resolution.

### Architecture Overview

```
Input: (batch, 23, 8, 8) - bitboard representation

Conv Block 1: 23 → 64 channels
  ├─ Conv 3×3 + BN + ReLU
  └─ Conv 3×3 + BN + ReLU

Conv Block 2: 64 → 128 channels
  ├─ Conv 3×3 + BN + ReLU
  └─ Conv 3×3 + BN + ReLU

Conv Block 3: 128 → 256 channels
  ├─ Conv 3×3 + BN + ReLU
  └─ Conv 3×3 + BN + ReLU

Conv Block 4: 256 → 256 channels
  ├─ Conv 3×3 + BN + ReLU
  ├─ Conv 3×3 + BN + ReLU
  └─ Dropout 0.3

Global Average Pool: (256, 8, 8) → (256,)

Value Head:
  ├─ FC 256 → 128 + ReLU + Dropout 0.5
  ├─ FC 128 → 32 + ReLU
  └─ FC 32 → 1

Output: (batch, 1) - evaluation score
```

### Model Statistics

**Default Configuration (base_channels=64):**
- Total parameters: **2,376,961** (~2.4M)
- Convolutional layers: ~2.3M params
- Fully connected layers: ~40K params

**Scalable:**
- `base_channels=32`: 600K params (lightweight)
- `base_channels=64`: 2.4M params (default)
- `base_channels=128`: 9.5M params (heavyweight)

### Key Features

1. **Maintains Spatial Resolution**: All conv blocks keep 8×8 size (padding=1 for 3×3 kernels)
2. **Progressive Depth**: Gradually increases channels (64→128→256→256)
3. **Regularization**: Dropout in last conv block and FC layers
4. **Batch Normalization**: After each conv for training stability
5. **Global Pooling**: Reduces spatial dimensions while preserving features
6. **He Initialization**: Proper weight initialization for ReLU networks

### Configuration Options

```python
model = ChessCNN(
    input_channels=23,      # 13/19/23 for different representations
    base_channels=64,       # Base channel multiplier
    dropout_conv=0.3,       # Dropout for conv layers
    dropout_fc=0.5          # Dropout for FC layers
)
```

### Usage Examples

**Basic usage:**
```python
from src.models import ChessCNN

model = ChessCNN(input_channels=23)
output = model(bitboard_tensor)  # (batch, 23, 8, 8) → (batch, 1)
```

**Factory function:**
```python
from src.models import create_chess_cnn

model = create_chess_cnn('full')  # Automatically sets input_channels=23
```

**Custom configuration:**
```python
model = ChessCNN(
    input_channels=23,
    base_channels=32,       # Smaller model
    dropout_conv=0.2,
    dropout_fc=0.4
)
```

### Design Rationale

**Why this architecture?**

1. **Two convs per block**: Deeper feature extraction without increasing parameters too much
2. **No pooling between blocks**: Chess board is only 8×8, pooling would lose spatial info
3. **Progressive channels**: More complex features need more channels
4. **Global avg pool**: Better than flatten+FC, more parameter efficient
5. **3-layer value head**: Gradually reduces dimensions 256→128→32→1

**Why not other designs?**

- **No max pooling**: Would reduce 8×8 to 4×4 too quickly, losing position information
- **No 1×1 convs**: Not needed since we're not doing dimension reduction mid-network
- **No residual connections**: Those are for ResNet variant (to be implemented)

### Variants Not Implemented (Yet)

See `docs/CNN_ARCHITECTURE.md` for 3 other architectural variants:
- Multi-Scale Feature CNN (multiple kernel sizes)
- Deep Narrow CNN (VGG-style)
- Inception-style CNN (parallel paths)

These can be implemented later for comparison experiments.

## Testing

Run tests:
```bash
python tests/test_model.py
```

Tests verify:
- Forward pass correctness
- All representations (13/19/23 channels)
- Component-level functionality
- Custom configurations
- Gradient flow

## Next Steps

To train this model:
1. Implement training loop (`src/training/trainer.py`)
2. Implement metrics tracking
3. Create experiment config
4. Run training on dataset

## Configuration Integration

Model configs defined in `src/config/model_configs.py`:

```python
from src.config import CNNConfig

config = CNNConfig(
    input_channels=23,
    base_channels=64,
    dropout_conv=0.3,
    dropout_fc=0.5
)
```

