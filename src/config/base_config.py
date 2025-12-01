"""
Configuration classes for experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    csv_path: str = "data/chess_data.csv"
    
    # Data loading
    max_samples: Optional[int] = None  # None = load all
    skip_header: bool = False
    
    # Data splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    shuffle: bool = True
    random_seed: int = 42
    
    # Preprocessing
    representation: str = 'full'  # 'simple', 'essential', or 'full'
    normalize_eval: bool = False  # Normalize evaluations to [-1, 1]
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_representations = ['simple', 'essential', 'full']
        if self.representation not in valid_representations:
            raise ValueError(
                f"representation must be one of {valid_representations}, "
                f"got '{self.representation}'"
            )
        
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Train, val, and test ratios must sum to 1.0, "
                f"got {self.train_ratio + self.val_ratio + self.test_ratio}"
            )


@dataclass
class ModelConfig:
    """Base configuration for model architecture."""
    
    # Model type
    model_type: str = 'resnet'  # 'cnn', 'resnet', etc.
    
    # Input configuration (determined by representation)
    input_channels: int = 23  # 13 for simple, 19 for essential, 23 for full
    
    # Output configuration
    output_size: int = 1  # Single evaluation score
    
    # Model-specific parameters (to be extended by subclasses)
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_num_parameters(self) -> Optional[int]:
        """
        Get the total number of trainable parameters.
        
        Returns:
            Number of parameters, or None if not yet determined
        """
        return None  # To be implemented by specific models


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Training hyperparameters
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Optimizer settings
    optimizer: str = 'adam'  # 'adam', 'adamw', 'sgd'
    betas: tuple = (0.9, 0.999)  # Adam/AdamW betas
    momentum: float = 0.9  # SGD momentum
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = 'reduce_on_plateau'  # 'reduce_on_plateau', 'cosine', 'step'
    scheduler_patience: int = 5  # For ReduceLROnPlateau
    scheduler_factor: float = 0.5  # For ReduceLROnPlateau
    min_lr: float = 1e-6
    
    # Loss function
    loss_function: str = 'mse'  # 'mse', 'mae', 'huber'
    huber_delta: float = 1.0  # For Huber loss
    
    # Regularization
    dropout: float = 0.0  # Dropout probability (if applicable)
    gradient_clip: Optional[float] = None  # Gradient clipping value
    
    # Training settings
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    
    # Device
    device: str = 'cuda'  # 'cuda' or 'cpu'
    
    # Checkpointing
    checkpoint_dir: str = "experiments/checkpoints"
    save_every_epoch: bool = True
    save_best_only: bool = False  # If True, only save when validation improves
    
    # Logging
    log_dir: str = "experiments/logs"
    log_interval: int = 10  # Log every N batches
    use_tensorboard: bool = False
    
    # Validation
    validate_every: int = 1  # Validate every N epochs
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4


@dataclass
class ExperimentConfig:
    """
    Top-level configuration combining all aspects of an experiment.
    
    This class brings together data, model, and training configurations
    to define a complete experimental setup.
    """
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    description: str = ""
    tags: list = field(default_factory=list)
    
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True  # Use deterministic algorithms when possible
    
    def __post_init__(self):
        """Validate and setup configuration."""
        # Ensure input channels match representation
        representation_to_channels = {
            'simple': 13,
            'essential': 19,
            'full': 23
        }
        expected_channels = representation_to_channels[self.data.representation]
        if self.model.input_channels != expected_channels:
            print(
                f"Warning: Adjusting model input_channels from {self.model.input_channels} "
                f"to {expected_channels} to match representation '{self.data.representation}'"
            )
            self.model.input_channels = expected_channels
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'tags': self.tags,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'random_seed': self.random_seed,
            'deterministic': self.deterministic,
        }
    
    def save(self, path: str):
        """
        Save configuration to a file.
        
        Args:
            path: Path to save the configuration
        """
        import json
        
        config_dict = self.to_dict()
        
        # Convert Path objects and other non-serializable objects
        def convert_value(v):
            if isinstance(v, Path):
                return str(v)
            elif isinstance(v, tuple):
                return list(v)
            return v
        
        def convert_dict(d):
            return {k: convert_value(v) if not isinstance(v, dict) 
                    else convert_dict(v) for k, v in d.items()}
        
        config_dict = convert_dict(config_dict)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """
        Load configuration from a file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            ExperimentConfig instance
        """
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct the configuration
        data_config = DataConfig(**config_dict['data'])
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        
        return cls(
            experiment_name=config_dict['experiment_name'],
            description=config_dict['description'],
            tags=config_dict['tags'],
            data=data_config,
            model=model_config,
            training=training_config,
            random_seed=config_dict['random_seed'],
            deterministic=config_dict['deterministic'],
        )

