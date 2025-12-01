"""
Convolutional Neural Network for chess position evaluation.

Progressive Depth CNN: Gradually increases channel depth while maintaining
spatial resolution, then collapses to evaluation score.
"""

import torch
import torch.nn as nn
from .base import ChessModel


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv → BatchNorm → ReLU.
    
    Two conv layers per block for deeper feature extraction.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        
        padding = kernel_size // 2  # Keep spatial size same
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class ChessCNN(ChessModel):
    """
    Progressive Depth CNN for chess evaluation.
    
    Architecture:
        - 4 convolutional blocks with increasing channels (64 → 128 → 256 → 256)
        - Maintains 8×8 spatial resolution throughout
        - Global average pooling
        - 3-layer value head
        
    Args:
        input_channels: Number of input channels (13/19/23 for different representations)
        base_channels: Base number of channels (default 64)
        dropout_conv: Dropout rate for convolutional layers
        dropout_fc: Dropout rate for fully connected layers
    """
    
    def __init__(
        self,
        input_channels: int = 23,
        base_channels: int = 64,
        dropout_conv: float = 0.3,
        dropout_fc: float = 0.5
    ):
        super().__init__(input_channels)
        
        self.base_channels = base_channels
        
        # Convolutional backbone
        # Block 1: Input → 64 channels
        self.block1 = ConvBlock(input_channels, base_channels)
        
        # Block 2: 64 → 128 channels
        self.block2 = ConvBlock(base_channels, base_channels * 2)
        
        # Block 3: 128 → 256 channels
        self.block3 = ConvBlock(base_channels * 2, base_channels * 4)
        
        # Block 4: 256 → 256 channels (with dropout)
        self.block4 = ConvBlock(base_channels * 4, base_channels * 4, dropout=dropout_conv)
        
        # Global pooling: (B, 256, 8, 8) → (B, 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Value head: 256 → 128 → 32 → 1
        fc_input = base_channels * 4
        self.fc1 = nn.Linear(fc_input, fc_input // 2)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_fc)
        
        self.fc2 = nn.Linear(fc_input // 2, 32)
        self.relu_fc2 = nn.ReLU(inplace=True)
        
        self.fc3 = nn.Linear(32, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, 8, 8)
            
        Returns:
            Evaluation scores (batch_size, 1)
        """
        # Convolutional blocks
        x = self.block1(x)  # (B, 64, 8, 8)
        x = self.block2(x)  # (B, 128, 8, 8)
        x = self.block3(x)  # (B, 256, 8, 8)
        x = self.block4(x)  # (B, 256, 8, 8)
        
        # Global pooling
        x = self.global_pool(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        
        # Value head
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu_fc2(x)
        
        x = self.fc3(x)  # (B, 1)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


def create_chess_cnn(representation='full', **kwargs):
    """
    Factory function to create ChessCNN with appropriate input channels.
    
    Args:
        representation: 'simple' (13ch), 'essential' (19ch), or 'full' (23ch)
        **kwargs: Additional arguments for ChessCNN
        
    Returns:
        ChessCNN model
    """
    channel_map = {'simple': 13, 'essential': 19, 'full': 23}
    input_channels = channel_map.get(representation, 23)
    
    return ChessCNN(input_channels=input_channels, **kwargs)


if __name__ == '__main__':
    # Test the model
    model = ChessCNN(input_channels=23)
    
    print("ChessCNN Architecture")
    print("=" * 60)
    print(f"Input channels: {model.input_channels}")
    print(f"Base channels: {model.base_channels}")
    print(f"Total parameters: {model.count_parameters():,}")
    print()
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 23, 8, 8)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print()
    
    # Test with different representations
    for rep in ['simple', 'essential', 'full']:
        model = create_chess_cnn(rep)
        print(f"{rep.capitalize()}: {model.input_channels} channels, {model.count_parameters():,} params")

