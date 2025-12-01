"""
Test script for CNN model.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import ChessCNN, create_chess_cnn
from src.utils import (
    print_header,
    print_section,
    print_success,
    print_error,
    print_info,
    print_metric_table,
    print_model_summary,
)


def test_cnn_architecture():
    """Test CNN model architecture and forward pass."""
    print_section("Testing ChessCNN Architecture")
    
    # Create model
    print_info("Creating CNN model...")
    model = ChessCNN(input_channels=23)
    
    # Print model summary
    print_model_summary(
        model_name="ChessCNN",
        parameters=model.count_parameters(),
        input_shape=(23, 8, 8),
        output_shape=(1,),
        additional_info={
            'Input Channels': model.input_channels,
            'Base Channels': model.base_channels,
        }
    )
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 23, 8, 8)
    
    print()
    print_info("Testing forward pass...")
    print(f"  Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print_success("Forward pass successful!")
    
    return model


def test_different_representations():
    """Test model with different input representations."""
    print_section("Testing Different Representations")
    
    representations = {
        'simple': 13,
        'essential': 19,
        'full': 23
    }
    
    results = {}
    for rep, channels in representations.items():
        model = create_chess_cnn(rep)
        
        # Test forward pass
        dummy_input = torch.randn(2, channels, 8, 8)
        with torch.no_grad():
            output = model(dummy_input)
        
        results[rep.capitalize()] = {
            'Input Channels': model.input_channels,
            'Parameters': f"{model.count_parameters():,}",
            'Output Shape': str(output.shape)
        }
        
        assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"
    
    print_metric_table(results['Simple'], title="Simple Representation")
    print_metric_table(results['Essential'], title="Essential Representation")
    print_metric_table(results['Full'], title="Full Representation")
    
    print_success("All representations working!")


def test_model_components():
    """Test individual model components."""
    print_section("Testing Model Components")
    
    model = ChessCNN(input_channels=23, base_channels=64)
    
    # Test each block
    x = torch.randn(2, 23, 8, 8)
    
    print_info(f"Input shape: {x.shape}")
    
    layers = [
        ('block1', (2, 64, 8, 8)),
        ('block2', (2, 128, 8, 8)),
        ('block3', (2, 256, 8, 8)),
        ('block4', (2, 256, 8, 8)),
    ]
    
    for layer_name, expected_shape in layers:
        x = getattr(model, layer_name)(x)
        if x.shape == expected_shape:
            print_success(f"{layer_name}: {x.shape}")
        else:
            print_error(f"{layer_name}: {x.shape} (expected {expected_shape})")
        assert x.shape == expected_shape
    
    x = model.global_pool(x)
    print_success(f"global_pool: {x.shape}")
    assert x.shape == (2, 256, 1, 1)
    
    x = x.view(x.size(0), -1)
    print_success(f"flatten: {x.shape}")
    assert x.shape == (2, 256)
    
    print()
    print_success("All components working correctly!")


def test_custom_config():
    """Test model with custom configuration."""
    print_section("Testing Custom Configuration")
    
    # Test with different base_channels
    configs = [
        {'base_channels': 32, 'dropout_conv': 0.2, 'dropout_fc': 0.3},
        {'base_channels': 64, 'dropout_conv': 0.3, 'dropout_fc': 0.5},
        {'base_channels': 128, 'dropout_conv': 0.4, 'dropout_fc': 0.6},
    ]
    
    results = {}
    for i, config in enumerate(configs):
        model = ChessCNN(input_channels=23, **config)
        params = model.count_parameters()
        
        config_name = f"Config {i+1} (base={config['base_channels']})"
        results[config_name] = {
            'Parameters': f"{params:,}",
            'Dropout Conv': config['dropout_conv'],
            'Dropout FC': config['dropout_fc'],
        }
        
        # Test forward
        x = torch.randn(2, 23, 8, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1)
        print_success(f"{config_name}: {params:,} parameters")
    
    print()
    print_success("Custom configurations working!")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print_section("Testing Gradient Flow")
    
    model = ChessCNN(input_channels=23)
    model.train()
    
    # Create dummy data
    print_info("Creating dummy data and computing gradients...")
    x = torch.randn(4, 23, 8, 8, requires_grad=True)
    target = torch.randn(4, 1)
    
    # Forward pass
    output = model(x)
    
    # Compute loss and backward
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Check gradients exist
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    grad_stats = {
        'Loss Value': f"{loss.item():.4f}",
        'Parameters with Gradients': f"{has_grad}/{total_params}",
        'Gradient Coverage': '100%' if has_grad == total_params else f"{has_grad/total_params*100:.1f}%"
    }
    print_metric_table(grad_stats, title="Gradient Flow Test")
    
    assert has_grad == total_params, "Not all parameters received gradients!"
    print_success("Gradients flowing correctly!")


if __name__ == '__main__':
    print_header("ChessCNN Model Test Suite")
    
    try:
        test_cnn_architecture()
        test_different_representations()
        test_model_components()
        test_custom_config()
        test_gradient_flow()
        
        print()
        print_header("All Tests Passed!")
        print_success("Model is ready for training")
        print_info("All components verified and working correctly")
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

