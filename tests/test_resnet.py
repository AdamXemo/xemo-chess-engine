"""
Tests for ResNet model.

Verifies model creation, forward pass, and output properties.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import (
    ChessResNet,
    ResidualBlock,
    create_chess_resnet,
    resnet_small,
    resnet_medium,
    resnet_large,
)
from src.config import ResNetConfig
from src.utils import (
    print_header,
    print_section,
    print_success,
    print_info,
    print_error,
    print_metric_table,
    print_model_summary,
)


def test_residual_block():
    """Test ResidualBlock creation and forward pass."""
    print_section("Testing Residual Block")
    
    block = ResidualBlock(num_filters=64)
    
    # Test forward pass
    x = torch.randn(4, 64, 8, 8)
    out = block(x)
    
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print_success(f"Input shape: {x.shape}")
    print_success(f"Output shape: {out.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in block.parameters())
    print_info(f"Block parameters: {params:,}")
    
    print_success("Residual block test passed!")


def test_resnet_creation():
    """Test ChessResNet model creation."""
    print_section("Testing ResNet Creation")
    
    # Test default creation
    model = ChessResNet()
    
    assert model.input_channels == 23
    assert model.num_blocks == 10
    assert model.num_filters == 128
    
    print_success(f"Default model created")
    print_info(f"  Blocks: {model.num_blocks}")
    print_info(f"  Filters: {model.num_filters}")
    print_info(f"  Parameters: {model.count_parameters():,}")
    
    # Test custom creation
    custom_model = ChessResNet(
        input_channels=13,
        num_blocks=5,
        num_filters=64,
        value_head_hidden=128
    )
    
    assert custom_model.input_channels == 13
    assert custom_model.num_blocks == 5
    assert custom_model.num_filters == 64
    
    print_success(f"Custom model created")
    print_info(f"  Blocks: {custom_model.num_blocks}")
    print_info(f"  Filters: {custom_model.num_filters}")
    print_info(f"  Parameters: {custom_model.count_parameters():,}")
    
    print_success("ResNet creation test passed!")


def test_resnet_forward():
    """Test ChessResNet forward pass."""
    print_section("Testing ResNet Forward Pass")
    
    model = resnet_medium()
    model.eval()
    
    # Test forward pass with different batch sizes
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 23, 8, 8)
        out = model(x)
        
        assert out.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {out.shape}"
        print_info(f"Batch {batch_size}: Input {x.shape} → Output {out.shape}")
    
    print_success("Forward pass test passed!")


def test_resnet_output_range():
    """Test that ResNet output is bounded by tanh to [-1, 1]."""
    print_section("Testing Output Range (tanh)")
    
    model = resnet_small()
    model.eval()
    
    # Test with random inputs
    x = torch.randn(100, 23, 8, 8) * 10  # Large values
    
    with torch.no_grad():
        out = model(x)
    
    min_val = out.min().item()
    max_val = out.max().item()
    
    assert min_val >= -1.0, f"Output min {min_val} < -1"
    assert max_val <= 1.0, f"Output max {max_val} > 1"
    
    print_info(f"Output range: [{min_val:.4f}, {max_val:.4f}]")
    print_success("Output range test passed (bounded to [-1, 1])!")


def test_resnet_presets():
    """Test preset ResNet configurations."""
    print_section("Testing ResNet Presets")
    
    presets = [
        ('small', resnet_small, 6, 64),
        ('medium', resnet_medium, 10, 128),
        ('large', resnet_large, 15, 128),
    ]
    
    preset_info = {}
    
    for name, factory_fn, expected_blocks, expected_filters in presets:
        model = factory_fn()
        
        assert model.num_blocks == expected_blocks, f"{name}: Expected {expected_blocks} blocks"
        assert model.num_filters == expected_filters, f"{name}: Expected {expected_filters} filters"
        
        params = model.count_parameters()
        preset_info[f"{name.capitalize()} ({expected_blocks}b/{expected_filters}f)"] = f"{params:,}"
    
    print_metric_table(preset_info, title="ResNet Presets")
    print_success("Preset tests passed!")


def test_create_chess_resnet():
    """Test factory function."""
    print_section("Testing Factory Function")
    
    # Test with different representations
    for rep, expected_channels in [('simple', 13), ('essential', 19), ('full', 23)]:
        model = create_chess_resnet(representation=rep, num_blocks=5, num_filters=64)
        
        assert model.input_channels == expected_channels
        print_info(f"{rep}: {expected_channels} input channels")
    
    print_success("Factory function test passed!")


def test_resnet_gradient_flow():
    """Test that gradients flow through residual connections."""
    print_section("Testing Gradient Flow")
    
    model = resnet_small()
    model.train()
    
    x = torch.randn(4, 23, 8, 8, requires_grad=True)
    target = torch.randn(4, 1)
    
    # Forward pass
    out = model(x)
    
    # Compute loss and backward
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "Input gradients should exist"
    
    # Check model gradients
    has_gradients = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    assert has_gradients, "All parameters should have gradients"
    
    print_info(f"Loss: {loss.item():.6f}")
    print_info(f"Input grad norm: {x.grad.norm().item():.6f}")
    print_success("Gradient flow test passed!")


def test_resnet_config():
    """Test ResNetConfig dataclass."""
    print_section("Testing ResNetConfig")
    
    config = ResNetConfig(
        input_channels=23,
        num_blocks=10,
        num_filters=128,
        value_head_hidden=256
    )
    
    assert config.model_type == 'resnet'
    assert config.num_blocks == 10
    assert config.num_filters == 128
    
    # Test parameter estimation
    estimated_params = config.get_num_parameters()
    
    # Create actual model and compare
    model = ChessResNet(
        input_channels=config.input_channels,
        num_blocks=config.num_blocks,
        num_filters=config.num_filters,
        value_head_hidden=config.value_head_hidden
    )
    actual_params = model.count_parameters()
    
    # Allow 20% tolerance
    assert abs(estimated_params - actual_params) / actual_params < 0.2, \
        f"Estimated {estimated_params:,} vs actual {actual_params:,}"
    
    print_info(f"Estimated parameters: {estimated_params:,}")
    print_info(f"Actual parameters: {actual_params:,}")
    print_success("ResNetConfig test passed!")


def test_model_comparison():
    """Compare CNN vs ResNet parameters."""
    print_section("Model Comparison")
    
    from src.models import ChessCNN
    
    # Create comparable models
    cnn = ChessCNN(input_channels=23, base_channels=64)
    resnet = resnet_medium()
    
    comparison = {
        'CNN (4 blocks, 64→256)': f"{cnn.count_parameters():,}",
        'ResNet Small (6b/64f)': f"{resnet_small().count_parameters():,}",
        'ResNet Medium (10b/128f)': f"{resnet_medium().count_parameters():,}",
        'ResNet Large (15b/128f)': f"{resnet_large().count_parameters():,}",
    }
    
    print_metric_table(comparison, title="Parameter Comparison")
    print_success("Model comparison complete!")


if __name__ == '__main__':
    print_header("ResNet Model Test Suite")
    
    try:
        test_residual_block()
        test_resnet_creation()
        test_resnet_forward()
        test_resnet_output_range()
        test_resnet_presets()
        test_create_chess_resnet()
        test_resnet_gradient_flow()
        test_resnet_config()
        test_model_comparison()
        
        print()
        print_header("All ResNet Tests Passed!")
        print_success("ResNet model is ready for training")
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

