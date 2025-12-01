"""
Base model class for chess evaluation networks.
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class ChessModel(ABC, nn.Module):
    """Abstract base class for all chess evaluation models."""
    
    def __init__(self, input_channels: int = 23):
        super().__init__()
        self.input_channels = input_channels
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, 8, 8)
            
        Returns:
            Evaluation tensor of shape (batch_size, 1)
        """
        pass
    
    def count_parameters(self):
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        """Get the device the model is on."""
        return next(self.parameters()).device

