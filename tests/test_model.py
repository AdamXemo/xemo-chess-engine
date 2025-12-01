"""
Test script for CNN model.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import ChessCNN, create_chess_cnn


def test_cnn_architecture():
    """Test CNN model architecture and forward pass."""
    print("=" * 60)
    print("Testing ChessCNN Architecture")
    print("=" * 60)
    
    # Create model
    model = ChessCNN(input_channels=23)
    
    print(f"\nModel Details:")
    print(f"  Input channels: {model.input_channels}")
    print(f"  Base channels: {model.base_channels}")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 23, 8, 8)
    
    print(f"\nForward Pass Test:")
    print(f"  Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.squeeze().tolist()}")
    
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print("  ✓ Forward pass successful!")
    
    return model


def test_different_representations():
    """Test model with different input representations."""
    print("\n" + "=" * 60)
    print("Testing Different Representations")
    print("=" * 60)
    
    representations = {
        'simple': 13,
        'essential': 19,
        'full': 23
    }
    
    for rep, channels in representations.items():
        model = create_chess_cnn(rep)
        
        # Test forward pass
        dummy_input = torch.randn(2, channels, 8, 8)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\n{rep.capitalize()} representation:")
        print(f"  Input channels: {model.input_channels}")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Output shape: {output.shape}")
        
        assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"
    
    print("\n  ✓ All representations working!")


def test_model_components():
    """Test individual model components."""
    print("\n" + "=" * 60)
    print("Testing Model Components")
    print("=" * 60)
    
    model = ChessCNN(input_channels=23, base_channels=64)
    
    # Test each block
    x = torch.randn(2, 23, 8, 8)
    
    print(f"\nInput: {x.shape}")
    
    x = model.block1(x)
    print(f"After block1: {x.shape} (expected: [2, 64, 8, 8])")
    assert x.shape == (2, 64, 8, 8)
    
    x = model.block2(x)
    print(f"After block2: {x.shape} (expected: [2, 128, 8, 8])")
    assert x.shape == (2, 128, 8, 8)
    
    x = model.block3(x)
    print(f"After block3: {x.shape} (expected: [2, 256, 8, 8])")
    assert x.shape == (2, 256, 8, 8)
    
    x = model.block4(x)
    print(f"After block4: {x.shape} (expected: [2, 256, 8, 8])")
    assert x.shape == (2, 256, 8, 8)
    
    x = model.global_pool(x)
    print(f"After global_pool: {x.shape} (expected: [2, 256, 1, 1])")
    assert x.shape == (2, 256, 1, 1)
    
    x = x.view(x.size(0), -1)
    print(f"After flatten: {x.shape} (expected: [2, 256])")
    assert x.shape == (2, 256)
    
    print("\n  ✓ All components working correctly!")


def test_custom_config():
    """Test model with custom configuration."""
    print("\n" + "=" * 60)
    print("Testing Custom Configuration")
    print("=" * 60)
    
    # Test with different base_channels
    configs = [
        {'base_channels': 32, 'dropout_conv': 0.2, 'dropout_fc': 0.3},
        {'base_channels': 64, 'dropout_conv': 0.3, 'dropout_fc': 0.5},
        {'base_channels': 128, 'dropout_conv': 0.4, 'dropout_fc': 0.6},
    ]
    
    for i, config in enumerate(configs):
        model = ChessCNN(input_channels=23, **config)
        params = model.count_parameters()
        
        print(f"\nConfig {i+1}: base_channels={config['base_channels']}")
        print(f"  Parameters: {params:,}")
        
        # Test forward
        x = torch.randn(2, 23, 8, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1)
    
    print("\n  ✓ Custom configurations working!")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    model = ChessCNN(input_channels=23)
    model.train()
    
    # Create dummy data
    x = torch.randn(4, 23, 8, 8, requires_grad=True)
    target = torch.randn(4, 1)
    
    # Forward pass
    output = model(x)
    
    # Compute loss and backward
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    print(f"\nLoss value: {loss.item():.4f}")
    
    # Check gradients exist
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    print(f"Parameters with gradients: {has_grad}/{total_params}")
    
    assert has_grad == total_params, "Not all parameters received gradients!"
    print("\n  ✓ Gradients flowing correctly!")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ChessCNN Model Test Suite")
    print("=" * 60)
    
    try:
        test_cnn_architecture()
        test_different_representations()
        test_model_components()
        test_custom_config()
        test_gradient_flow()
        
        print("\n" + "=" * 60)
        print("All Tests Passed! ✓")
        print("=" * 60)
        print("\nModel is ready for training!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

