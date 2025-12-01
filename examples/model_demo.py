"""
Demo script showing how to use the CNN model with actual chess data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models import ChessCNN, create_chess_cnn
from src.data import ChessDataLoader
from src.config import CNNConfig


def demo_basic_model():
    """Demo: Create and use the model."""
    print("=" * 60)
    print("Demo 1: Basic Model Creation")
    print("=" * 60)
    
    # Create model
    model = ChessCNN(input_channels=23)
    
    print(f"\nModel created:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {model.get_device()}")
    
    # Create dummy input
    dummy_board = torch.randn(1, 23, 8, 8)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        prediction = model(dummy_board)
    
    print(f"\nDummy prediction: {prediction.item():.2f}")
    print("  (Random weights, so prediction is meaningless)")


def demo_with_real_data():
    """Demo: Use model with actual chess positions."""
    print("\n" + "=" * 60)
    print("Demo 2: Model with Real Chess Data")
    print("=" * 60)
    
    # Load small dataset
    print("\nLoading dataset...")
    train_ds, val_ds, _ = ChessDataLoader.create_datasets(
        csv_path='data/chess_data.csv',
        representation='full',
        max_samples=1000,
        random_seed=42
    )
    
    # Create dataloader
    loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    
    # Get one batch
    bitboards, evaluations = next(iter(loader))
    
    print(f"\nBatch loaded:")
    print(f"  Bitboards shape: {bitboards.shape}")
    print(f"  Evaluations shape: {evaluations.shape}")
    print(f"  True evaluations: {evaluations.squeeze().tolist()}")
    
    # Create model and predict
    model = ChessCNN(input_channels=23)
    model.eval()
    
    with torch.no_grad():
        predictions = model(bitboards)
    
    print(f"\nModel predictions (untrained):")
    print(f"  {predictions.squeeze().tolist()}")
    print("\n  Note: Random predictions since model is untrained")


def demo_different_configurations():
    """Demo: Different model configurations."""
    print("\n" + "=" * 60)
    print("Demo 3: Different Model Configurations")
    print("=" * 60)
    
    configs = [
        {'base_channels': 32, 'name': 'Small'},
        {'base_channels': 64, 'name': 'Default'},
        {'base_channels': 128, 'name': 'Large'},
    ]
    
    for cfg in configs:
        model = ChessCNN(input_channels=23, base_channels=cfg['base_channels'])
        params = model.count_parameters()
        
        print(f"\n{cfg['name']} model (base_channels={cfg['base_channels']}):")
        print(f"  Parameters: {params:,}")
        print(f"  Memory: ~{params * 4 / 1e6:.1f}MB (fp32)")


def demo_model_inference_speed():
    """Demo: Measure inference speed."""
    print("\n" + "=" * 60)
    print("Demo 4: Inference Speed Test")
    print("=" * 60)
    
    model = ChessCNN(input_channels=23)
    model.eval()
    
    # Warm up
    dummy = torch.randn(32, 23, 8, 8)
    with torch.no_grad():
        _ = model(dummy)
    
    # Time it
    import time
    
    batch_sizes = [1, 16, 64, 256]
    
    print("\nInference times (CPU):")
    for bs in batch_sizes:
        dummy = torch.randn(bs, 23, 8, 8)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy)
        elapsed = (time.time() - start) / 10
        
        per_position = elapsed / bs * 1000
        
        print(f"  Batch size {bs:3d}: {elapsed*1000:6.2f}ms total, {per_position:.2f}ms/position")


def demo_model_on_gpu():
    """Demo: Use model on GPU if available."""
    print("\n" + "=" * 60)
    print("Demo 5: GPU Usage")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nGPU available: {torch.cuda.get_device_name(0)}")
        
        # Move model to GPU
        model = ChessCNN(input_channels=23).to(device)
        print(f"Model device: {model.get_device()}")
        
        # Create data on GPU
        dummy = torch.randn(64, 23, 8, 8).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        
        print(f"Output device: {output.device}")
        print("  ✓ GPU inference working!")
        
    else:
        print("\nNo GPU available. Model will run on CPU.")


def demo_save_load_model():
    """Demo: Save and load model."""
    print("\n" + "=" * 60)
    print("Demo 6: Save and Load Model")
    print("=" * 60)
    
    # Create and save
    model = ChessCNN(input_channels=23, base_channels=64)
    save_path = '/tmp/chess_cnn_demo.pth'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_channels': model.input_channels,
        'base_channels': model.base_channels,
    }, save_path)
    
    print(f"\nModel saved to: {save_path}")
    
    # Load
    checkpoint = torch.load(save_path)
    
    new_model = ChessCNN(
        input_channels=checkpoint['input_channels'],
        base_channels=checkpoint['base_channels']
    )
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    print(f"  Parameters: {new_model.count_parameters():,}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ChessCNN Model Demo")
    print("=" * 60)
    
    demo_basic_model()
    demo_with_real_data()
    demo_different_configurations()
    demo_model_inference_speed()
    demo_model_on_gpu()
    demo_save_load_model()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext: Implement training loop to actually train the model!")

