"""Training configuration."""

from typing import Optional, Dict, Any, List
from pathlib import Path

from .base import BaseConfig
from .data import DataConfig
from .model import ModelConfig
from ..utils import (
    validate_positive,
    validate_range,
    validate_path,
    ValidationError,
    get_logger
)

logger = get_logger(__name__)


class TrainingConfig(BaseConfig):
    """
    Configuration for model training.
    
    Attributes:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        optimizer: Optimizer name or configuration
        loss: Loss function name
        metrics: List of metrics to track
        callbacks: Callback configuration
        seed: Random seed for reproducibility
        mixed_precision: Use mixed precision training
        output_dir: Directory for saving outputs
        
    Example:
        >>> config = TrainingConfig(
        ...     epochs=50,
        ...     batch_size=32,
        ...     learning_rate=0.001,
        ...     seed=42
        ... )
    """
    
    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: Optional[List[str]] = None,
        callbacks: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        mixed_precision: bool = False,
        output_dir: str = 'outputs',
        experiment_name: Optional[str] = None,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[ModelConfig] = None,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        checkpoint_monitor: str = 'val_accuracy',
        checkpoint_mode: str = 'max',
        verbose: int = 1
    ):
        """
        Initialize training configuration.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Initial learning rate
            optimizer: Optimizer name
            loss: Loss function name
            metrics: List of metrics
            callbacks: Callback configuration dictionary
            seed: Random seed
            mixed_precision: Enable mixed precision training
            output_dir: Output directory path
            experiment_name: Name of experiment
            data_config: Data configuration
            model_config: Model configuration
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for reducing learning rate
            reduce_lr_factor: Factor for reducing learning rate
            checkpoint_monitor: Metric to monitor for checkpointing
            checkpoint_mode: Mode for checkpointing ('min' or 'max')
            verbose: Verbosity level
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or ['accuracy']
        self.callbacks = callbacks or self._default_callbacks()
        self.seed = seed
        self.mixed_precision = mixed_precision
        self.output_dir = output_dir
        self.experiment_name = experiment_name or 'default_experiment'
        
        # Nested configs
        self.data_config = data_config
        self.model_config = model_config
        
        # Callback parameters
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.verbose = verbose
        
        super().__init__()
        logger.info(f"TrainingConfig created for experiment: {self.experiment_name}")
    
    @staticmethod
    def _default_callbacks() -> Dict[str, Any]:
        """Get default callback configuration."""
        return {
            'early_stopping': True,
            'checkpoint': True,
            'reduce_lr': True,
            'tensorboard': False,
            'csv_logger': True
        }
    
    def validate(self) -> bool:
        """
        Validate training configuration.
        
        Returns:
            True if valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate epochs
        validate_positive(self.epochs)
        
        # Validate batch size
        validate_positive(self.batch_size)
        
        # Validate learning rate
        validate_range(self.learning_rate, min_val=1e-10, max_val=1.0)
        
        # Validate seed
        if self.seed < 0:
            raise ValidationError("Seed must be non-negative")
        
        # Validate output dir parent exists
        output_path = Path(self.output_dir)
        if output_path.parent != Path('.'):
            validate_path(output_path.parent, must_exist=True, path_type='dir')
        
        # Validate patience values
        validate_positive(self.early_stopping_patience)
        validate_positive(self.reduce_lr_patience)
        
        # Validate reduce_lr_factor
        validate_range(self.reduce_lr_factor, min_val=0.01, max_val=0.99)
        
        # Validate checkpoint mode
        if self.checkpoint_mode not in ['min', 'max']:
            raise ValidationError("Checkpoint mode must be 'min' or 'max'")
        
        # Validate nested configs if present
        if self.data_config is not None:
            self.data_config.validate()
        
        if self.model_config is not None:
            self.model_config.validate()
        
        self._validated = True
        logger.info("TrainingConfig validation passed")
        return True
    
    def get_output_path(self, subdir: str = '') -> Path:
        """
        Get path for outputs.
        
        Args:
            subdir: Subdirectory name
            
        Returns:
            Full output path
        """
        base_path = Path(self.output_dir) / self.experiment_name
        if subdir:
            return base_path / subdir
        return base_path
    
    def enable_callback(self, callback_name: str) -> None:
        """Enable a specific callback."""
        self.callbacks[callback_name] = True
        logger.debug(f"Enabled callback: {callback_name}")
    
    def disable_callback(self, callback_name: str) -> None:
        """Disable a specific callback."""
        self.callbacks[callback_name] = False
        logger.debug(f"Disabled callback: {callback_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling nested configs."""
        config_dict = super().to_dict()
        
        # Handle nested configs
        if self.data_config is not None:
            config_dict['data_config'] = self.data_config.to_dict()
        
        if self.model_config is not None:
            config_dict['model_config'] = self.model_config.to_dict()
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary."""
        # Handle nested configs
        if 'data_config' in config_dict and isinstance(config_dict['data_config'], dict):
            config_dict['data_config'] = DataConfig.from_dict(config_dict['data_config'])
        
        if 'model_config' in config_dict and isinstance(config_dict['model_config'], dict):
            config_dict['model_config'] = ModelConfig.from_dict(config_dict['model_config'])
        
        return cls(**config_dict)
    
    @classmethod
    def quick_start(
        cls,
        dataset_path: str,
        architecture: str = 'resnet50',
        num_classes: int = 4,
        epochs: int = 50,
        **kwargs
    ) -> 'TrainingConfig':
        """
        Create a quick-start configuration with sensible defaults.
        
        Args:
            dataset_path: Path to dataset
            architecture: Model architecture
            num_classes: Number of classes
            epochs: Number of epochs
            **kwargs: Additional configuration parameters
            
        Returns:
            Complete TrainingConfig with nested configs
            
        Example:
            >>> config = TrainingConfig.quick_start(
            ...     dataset_path='/path/to/data',
            ...     architecture='resnet50',
            ...     num_classes=4
            ... )
        """
        # Create data config
        data_config = DataConfig(dataset_path=dataset_path)
        
        # Create model config
        model_config = ModelConfig.for_architecture(architecture, num_classes)
        
        # Create training config
        return cls(
            epochs=epochs,
            batch_size=32,
            data_config=data_config,
            model_config=model_config,
            **kwargs
        )
