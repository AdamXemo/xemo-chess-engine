"""
Configuration management for chess neural network experiments.
"""

from .base_config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
)
from .model_configs import (
    CNNConfig,
    ResNetConfig,
    get_model_config,
)

__all__ = [
    'DataConfig',
    'ModelConfig',
    'TrainingConfig',
    'ExperimentConfig',
    'CNNConfig',
    'ResNetConfig',
    'get_model_config',
]

