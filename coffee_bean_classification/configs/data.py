"""Data configuration."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..utils import (
    ValidationError,
    get_logger,
    validate_image_size,
    validate_path,
    validate_positive,
    validate_ratio,
)
from .base import BaseConfig

logger = get_logger(__name__)


class DataConfig(BaseConfig):
    """
    Configuration for data pipeline.

    Attributes:
        dataset_path: Path to dataset directory
        image_size: Image size as (height, width) or single int
        batch_size: Batch size for training
        split_ratio: Train/val/test split ratios (must sum to 1.0)
        augmentation_params: Data augmentation parameters
        class_names: List of class names
        cache_dir: Directory for caching processed data
        num_parallel_calls: Number of parallel calls for data loading

    Example:
        >>> config = DataConfig(
        ...     dataset_path='/path/to/dataset',
        ...     image_size=[224, 224],
        ...     batch_size=32,
        ...     split_ratio=(0.7, 0.15, 0.15)
        ... )
    """

    def __init__(
        self,
        dataset_path: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        augmentation_params: Optional[Dict[str, Any]] = None,
        class_names: Optional[list] = None,
        cache_dir: Optional[str] = None,
        num_parallel_calls: int = -1,
        shuffle_buffer_size: int = 1000,
        prefetch_buffer_size: int = -1,
    ):
        """
        Initialize data configuration.

        Args:
            dataset_path: Path to dataset directory
            image_size: Target image size
            batch_size: Training batch size
            split_ratio: (train, val, test) split ratios
            augmentation_params: Augmentation configuration
            class_names: List of class names (auto-detected if None)
            cache_dir: Directory for caching (None to disable)
            num_parallel_calls: Parallel calls for data loading (-1 for AUTOTUNE)
            shuffle_buffer_size: Buffer size for shuffling
            prefetch_buffer_size: Buffer size for prefetching (-1 for AUTOTUNE)
        """
        self.dataset_path = dataset_path
        self.image_size = validate_image_size(image_size)
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.augmentation_params = augmentation_params or self._default_augmentation()
        self.class_names = class_names
        self.cache_dir = cache_dir
        self.num_parallel_calls = num_parallel_calls
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_buffer_size = prefetch_buffer_size

        super().__init__()
        logger.info(f"DataConfig created for {dataset_path}")

    @staticmethod
    def _default_augmentation() -> Dict[str, Any]:
        """Get default augmentation parameters."""
        return {
            "horizontal_flip": True,
            "rotation_range": 0.2,
            "zoom_range": 0.1,
            "brightness_range": (0.8, 1.2),
            "contrast_range": (0.8, 1.2),
        }

    def validate(self) -> bool:
        """
        Validate data configuration.

        Returns:
            True if valid

        Raises:
            ValidationError: If configuration is invalid
        """
        # Validate dataset path
        validate_path(self.dataset_path, must_exist=True, path_type="dir")

        # Validate batch size
        validate_positive(self.batch_size)

        # Validate split ratio
        validate_ratio(self.split_ratio, total=1.0)

        # Validate image size
        if self.image_size[0] <= 0 or self.image_size[1] <= 0:
            raise ValidationError("Image size must be positive")

        # Validate cache dir if provided
        if self.cache_dir is not None:
            validate_path(Path(self.cache_dir).parent, must_exist=True, path_type="dir")

        self._validated = True
        logger.info("DataConfig validation passed")
        return True

    @property
    def train_ratio(self) -> float:
        """Get training split ratio."""
        return self.split_ratio[0]

    @property
    def val_ratio(self) -> float:
        """Get validation split ratio."""
        return self.split_ratio[1]

    @property
    def test_ratio(self) -> float:
        """Get test split ratio."""
        return self.split_ratio[2]

    def get_num_classes(self) -> Optional[int]:
        """Get number of classes."""
        if self.class_names is not None:
            return len(self.class_names)
        return None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        """Create DataConfig from dictionary."""
        # Convert image_size if it's a list
        if "image_size" in config_dict:
            img_size = config_dict["image_size"]
            if isinstance(img_size, list):
                config_dict["image_size"] = tuple(img_size)

        # Convert split_ratio if it's a list
        if "split_ratio" in config_dict:
            split = config_dict["split_ratio"]
            if isinstance(split, list):
                config_dict["split_ratio"] = tuple(split)

        return cls(**config_dict)
