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
    """
    Configuration for ResNet model.
    
    Based on AlphaZero architecture with residual blocks and
    tanh output for bounded evaluation predictions.
    """
    
    model_type: str = 'resnet'
    
    # Architecture parameters
    num_blocks: int = 10  # Number of residual blocks
    num_filters: int = 128  # Filters per layer (constant throughout)
    value_head_hidden: int = 256  # Hidden units in value head FC
    
    def __post_init__(self):
        """Update model_params dict."""
        self.model_params = {
            'num_blocks': self.num_blocks,
            'num_filters': self.num_filters,
            'value_head_hidden': self.value_head_hidden,
        }
    
    def get_num_parameters(self) -> int:
        """Estimate number of parameters (approximate)."""
        # Each ResBlock: 2 * (3*3 * F * F) = 18 * F^2
        # Initial conv: 3*3 * 23 * F
        # Value head: F + 64 * 256 + 256
        f = self.num_filters
        n = self.num_blocks
        params = (
            9 * 23 * f +           # Initial conv
            n * 2 * 9 * f * f +    # Residual blocks
            f +                     # Value conv
            64 * self.value_head_hidden +  # FC1
            self.value_head_hidden  # FC2
        )
        return int(params * 1.05)  # ~5% for biases/BN


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

