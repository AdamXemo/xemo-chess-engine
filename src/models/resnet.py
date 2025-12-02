"""
Residual Neural Network for chess position evaluation.

Based on AlphaZero architecture with skip connections for training
deeper networks without vanishing gradients.
"""

import torch
import torch.nn as nn
from .base import ChessModel


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    
    Architecture:
        x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
    
    The skip connection adds the input directly to the output,
    allowing gradients to flow through without degradation.
    """
    
    def __init__(self, num_filters: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
        self.relu_out = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        out = out + identity
        out = self.relu_out(out)
        
        return out


class ChessResNet(ChessModel):
    """
    Residual CNN for chess evaluation based on AlphaZero architecture.
    
    Architecture:
        - Initial 3×3 conv layer
        - N residual blocks with skip connections
        - AlphaZero-style value head with tanh output
        
    The tanh output bounds predictions to [-1, 1], matching normalized
    evaluation targets.
    
    Args:
        input_channels: Number of input channels (13/19/23)
        num_blocks: Number of residual blocks (default 10)
        num_filters: Number of filters in each layer (default 128)
        value_head_hidden: Hidden units in value head FC layer (default 256)
    """
    
    def __init__(
        self,
        input_channels: int = 23,
        num_blocks: int = 10,
        num_filters: int = 128,
        value_head_hidden: int = 256
    ):
        super().__init__(input_channels)
        
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.value_head_hidden = value_head_hidden
        
        # Initial convolution: input_channels → num_filters
        self.initial_conv = nn.Conv2d(
            input_channels, num_filters, 
            kernel_size=3, padding=1, bias=False
        )
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])
        
        # Value head (AlphaZero style)
        # 1×1 conv to reduce channels to 1
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        
        # Flatten 8×8 = 64 values, then FC layers
        self.value_fc1 = nn.Linear(64, value_head_hidden)
        self.value_fc1_relu = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(value_head_hidden, 1)
        
        # tanh for bounded output [-1, 1]
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, 8, 8)
            
        Returns:
            Evaluation scores in [-1, 1] (batch_size, 1)
        """
        # Initial conv
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Value head
        v = self.value_conv(x)      # (B, 1, 8, 8)
        # Clip extreme values to prevent numerical instability
        v = torch.clamp(v, min=-100, max=100)
        v = self.value_bn(v)
        v = self.value_relu(v)
        
        v = v.view(v.size(0), -1)   # (B, 64)
        v = self.value_fc1(v)
        v = self.value_fc1_relu(v)
        v = self.value_fc2(v)       # (B, 1)
        v = self.tanh(v)            # [-1, 1]
        
        return v
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


def create_chess_resnet(
    representation: str = 'full',
    num_blocks: int = 10,
    num_filters: int = 128,
    **kwargs
) -> ChessResNet:
    """
    Factory function to create ChessResNet with appropriate input channels.
    
    Args:
        representation: 'simple' (13ch), 'essential' (19ch), or 'full' (23ch)
        num_blocks: Number of residual blocks
        num_filters: Number of filters per layer
        **kwargs: Additional arguments for ChessResNet
        
    Returns:
        ChessResNet model
    """
    channel_map = {'simple': 13, 'essential': 19, 'full': 23}
    input_channels = channel_map.get(representation, 23)
    
    return ChessResNet(
        input_channels=input_channels,
        num_blocks=num_blocks,
        num_filters=num_filters,
        **kwargs
    )


# Preset configurations
def resnet_small() -> ChessResNet:
    """Small ResNet: 6 blocks, 64 filters (~3-4M params)"""
    return ChessResNet(num_blocks=6, num_filters=64)


def resnet_medium() -> ChessResNet:
    """Medium ResNet: 10 blocks, 128 filters (~8-10M params)"""
    return ChessResNet(num_blocks=10, num_filters=128)


def resnet_large() -> ChessResNet:
    """Large ResNet: 15 blocks, 128 filters (~15M params)"""
    return ChessResNet(num_blocks=15, num_filters=128)


if __name__ == '__main__':
    # Test the model
    print("ChessResNet Architecture Test")
    print("=" * 60)
    
    for name, model_fn in [('small', resnet_small), ('medium', resnet_medium), ('large', resnet_large)]:
        model = model_fn()
        print(f"\n{name.upper()} ResNet:")
        print(f"  Blocks: {model.num_blocks}")
        print(f"  Filters: {model.num_filters}")
        print(f"  Parameters: {model.count_parameters():,}")
        
        # Test forward pass
        batch_size = 4
        dummy_input = torch.randn(batch_size, 23, 8, 8)
        output = model(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

