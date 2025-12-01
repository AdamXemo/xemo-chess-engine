"""
Model-specific configuration classes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from .base_config import ModelConfig


@dataclass
class CNNConfig(ModelConfig):
    """Configuration for CNN model."""
    
    model_type: str = 'cnn'
    
    # Architecture parameters
    base_channels: int = 64  # Base number of channels (multiplied in deeper layers)
    dropout_conv: float = 0.3  # Dropout for convolutional layers
    dropout_fc: float = 0.5  # Dropout for fully connected layers
    
    def __post_init__(self):
        """Update model_params dict."""
        self.model_params = {
            'base_channels': self.base_channels,
            'dropout_conv': self.dropout_conv,
            'dropout_fc': self.dropout_fc,
        }
    
    def get_num_parameters(self) -> int:
        """Estimate number of parameters (approximate)."""
        # Rough estimate based on base_channels=64:
        # Conv blocks: ~550K, FC layers: ~50K
        return int(600000 * (self.base_channels / 64) ** 2)


@dataclass
class ResNetConfig(ModelConfig):
    """Configuration for ResNet model (to be implemented)."""
    
    model_type: str = 'resnet'
    
    # Architecture parameters
    num_blocks: int = 10  # Number of residual blocks
    num_channels: int = 256  # Channels in residual blocks
    dropout: float = 0.3
    
    def __post_init__(self):
        """Update model_params dict."""
        self.model_params = {
            'num_blocks': self.num_blocks,
            'num_channels': self.num_channels,
            'dropout': self.dropout,
        }


def get_model_config(model_type: str, **kwargs) -> ModelConfig:
    """
    Factory function to create model config based on type.
    
    Args:
        model_type: 'cnn', 'resnet', etc.
        **kwargs: Additional configuration parameters
        
    Returns:
        Appropriate ModelConfig subclass
    """
    config_map = {
        'cnn': CNNConfig,
        'resnet': ResNetConfig,
    }
    
    config_class = config_map.get(model_type, ModelConfig)
    return config_class(model_type=model_type, **kwargs)

