"""Model configuration."""

from typing import List, Optional, Tuple

from ..utils import ValidationError, get_logger, validate_choice, validate_positive, validate_range
from .base import BaseConfig

logger = get_logger(__name__)


class ModelConfig(BaseConfig):
    """
    Configuration for model architecture.

    Attributes:
        architecture: Model architecture name
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        weights: Pre-trained weights ('imagenet' or None)
        dropout_rate: Dropout rate
        freeze_backbone: Whether to freeze backbone during initial training
        pooling: Global pooling type ('avg' or 'max')

    Example:
        >>> config = ModelConfig(
        ...     architecture='resnet50',
        ...     input_shape=(224, 224, 3),
        ...     num_classes=4,
        ...     weights='imagenet'
        ... )
    """

    SUPPORTED_ARCHITECTURES = [
        "resnet50",
        "efficientnet_b0",
        "efficientnet_b3",
        "mobilenet_v3",
        "densenet121",
    ]

    def __init__(
        self,
        architecture: str,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 4,
        weights: Optional[str] = "imagenet",
        dropout_rate: float = 0.2,
        freeze_backbone: bool = True,
        pooling: str = "avg",
        activation: str = "softmax",
        optimizer: str = "adam",
        loss: str = "categorical_crossentropy",
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialize model configuration.

        Args:
            architecture: Model architecture name
            input_shape: Input tensor shape
            num_classes: Number of output classes
            weights: Pre-trained weights source
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Freeze backbone weights
            pooling: Global pooling type
            activation: Output activation function
            optimizer: Optimizer name
            loss: Loss function name
            metrics: List of metric names
        """
        self.architecture = architecture.lower()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self.dropout_rate = dropout_rate
        self.freeze_backbone = freeze_backbone
        self.pooling = pooling
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or ["accuracy"]

        super().__init__()
        logger.info(f"ModelConfig created for {architecture}")

    def validate(self) -> bool:
        """
        Validate model configuration.

        Returns:
            True if valid

        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate architecture
        validate_choice(self.architecture, self.SUPPORTED_ARCHITECTURES)

        # Validate input shape
        if len(self.input_shape) != 3:
            raise ValidationError("Input shape must be (height, width, channels)")

        for dim in self.input_shape:
            validate_positive(dim)

        # Validate num_classes
        validate_positive(self.num_classes)

        # Validate dropout rate
        validate_range(self.dropout_rate, min_val=0.0, max_val=1.0)

        # Validate pooling
        validate_choice(self.pooling, ["avg", "max", None])

        # Validate weights
        if self.weights is not None:
            validate_choice(self.weights, ["imagenet"])

        self._validated = True
        logger.info("ModelConfig validation passed")
        return True

    def get_input_size(self) -> Tuple[int, int]:
        """Get input image size (height, width)."""
        return (self.input_shape[0], self.input_shape[1])

    def get_channels(self) -> int:
        """Get number of input channels."""
        return self.input_shape[2]

    def is_transfer_learning(self) -> bool:
        """Check if using transfer learning."""
        return self.weights is not None

    @classmethod
    def from_dict(cls, config_dict) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        # Convert input_shape if it's a list
        if "input_shape" in config_dict:
            input_shape = config_dict["input_shape"]
            if isinstance(input_shape, list):
                config_dict["input_shape"] = tuple(input_shape)

        return cls(**config_dict)

    @classmethod
    def for_architecture(cls, architecture: str, num_classes: int = 4, **kwargs) -> "ModelConfig":
        """
        Create configuration for specific architecture with optimal settings.

        Args:
            architecture: Model architecture name
            num_classes: Number of classes
            **kwargs: Additional configuration parameters

        Returns:
            Optimized ModelConfig for the architecture
        """
        # Architecture-specific optimal settings
        settings = {
            "resnet50": {
                "input_shape": (224, 224, 3),
                "dropout_rate": 0.3,
            },
            "efficientnet_b0": {
                "input_shape": (224, 224, 3),
                "dropout_rate": 0.2,
            },
            "efficientnet_b3": {
                "input_shape": (300, 300, 3),
                "dropout_rate": 0.3,
            },
            "mobilenet_v3": {
                "input_shape": (224, 224, 3),
                "dropout_rate": 0.2,
            },
            "densenet121": {
                "input_shape": (224, 224, 3),
                "dropout_rate": 0.3,
            },
        }

        arch_settings = settings.get(architecture.lower(), {})
        arch_settings.update(kwargs)

        return cls(architecture=architecture, num_classes=num_classes, **arch_settings)
