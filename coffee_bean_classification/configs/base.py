"""Base configuration class for all configurations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union
import copy

from ..utils import save_json, load_json, save_yaml, load_yaml, get_logger

logger = get_logger(__name__)


class BaseConfig(ABC):
    """
    Abstract base class for all configuration classes.

    All configuration classes must inherit from this base class and implement
    the validate() method.

    Attributes:
        _validated: Whether the configuration has been validated

    Example:
        >>> class MyConfig(BaseConfig):
        ...     def __init__(self, param1, param2):
        ...         self.param1 = param1
        ...         self.param2 = param2
        ...         super().__init__()
        ...
        ...     def validate(self) -> bool:
        ...         # Validation logic
        ...         return True
    """

    def __init__(self):
        """Initialize base configuration."""
        self._validated = False
        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate configuration parameters.

        This method must be implemented by all subclasses to validate
        their specific configuration parameters.

        Returns:
            True if configuration is valid

        Raises:
            ValidationError: If configuration is invalid
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> config = MyConfig(param1=10, param2='value')
            >>> config_dict = config.to_dict()
        """
        # Get all public attributes (not starting with _)
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                # Handle nested configs
                if isinstance(value, BaseConfig):
                    config_dict[key] = value.to_dict()
                else:
                    config_dict[key] = value

        logger.debug(f"Converted {self.__class__.__name__} to dict")
        return config_dict

    def save(self, path: Union[str, Path], format: str = "auto") -> Path:
        """
        Save configuration to file.

        Args:
            path: Output file path
            format: File format ('json', 'yaml', or 'auto' to infer from extension)

        Returns:
            Path to saved file

        Raises:
            ValueError: If format is unknown

        Example:
            >>> config.save('config.yaml')
            >>> config.save('config.json', format='json')
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif suffix == ".json":
                format = "json"
            else:
                raise ValueError(
                    f"Cannot infer format from extension '{suffix}'. "
                    "Please specify format explicitly."
                )

        # Save based on format
        config_dict = self.to_dict()

        if format == "json":
            save_json(config_dict, path)
        elif format == "yaml":
            save_yaml(config_dict, path)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {self.__class__.__name__} to {path}")
        return path

    @classmethod
    def load(cls, path: Union[str, Path], format: str = "auto") -> "BaseConfig":
        """
        Load configuration from file.

        Args:
            path: Input file path
            format: File format ('json', 'yaml', or 'auto' to infer from extension)

        Returns:
            Configuration instance

        Raises:
            ValueError: If format is unknown
            FileNotFoundError: If file doesn't exist

        Example:
            >>> config = MyConfig.load('config.yaml')
            >>> config = MyConfig.load('config.json', format='json')
        """
        path = Path(path)

        # Auto-detect format
        if format == "auto":
            suffix = path.suffix.lower()
            if suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif suffix == ".json":
                format = "json"
            else:
                raise ValueError(
                    f"Cannot infer format from extension '{suffix}'. "
                    "Please specify format explicitly."
                )

        # Load based on format
        if format == "json":
            config_dict = load_json(path)
        elif format == "yaml":
            config_dict = load_yaml(path)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Create instance from dictionary
        instance = cls.from_dict(config_dict)

        logger.info(f"Loaded {cls.__name__} from {path}")
        return instance

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """
        Create configuration instance from dictionary.

        This method should be overridden by subclasses if they need
        custom deserialization logic.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration instance

        Example:
            >>> config_dict = {'param1': 10, 'param2': 'value'}
            >>> config = MyConfig.from_dict(config_dict)
        """
        # Default implementation: pass dict as kwargs
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "BaseConfig":
        """
        Load configuration from YAML file.

        Args:
            path: YAML file path

        Returns:
            Configuration instance
        """
        return cls.load(path, format="yaml")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BaseConfig":
        """
        Load configuration from JSON file.

        Args:
            path: JSON file path

        Returns:
            Configuration instance
        """
        return cls.load(path, format="json")

    def copy(self) -> "BaseConfig":
        """
        Create a deep copy of the configuration.

        Returns:
            Deep copy of configuration

        Example:
            >>> config_copy = config.copy()
            >>> config_copy.param1 = 20  # Doesn't affect original
        """
        return copy.deepcopy(self)

    def update(self, **kwargs) -> "BaseConfig":
        """
        Update configuration parameters.

        Args:
            **kwargs: Parameters to update

        Returns:
            Self (for method chaining)

        Example:
            >>> config.update(epochs=100, batch_size=64)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated {key} = {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")

        # Invalidate validation
        self._validated = False
        return self

    def merge(self, other: "BaseConfig") -> "BaseConfig":
        """
        Merge another configuration into this one.

        Args:
            other: Configuration to merge

        Returns:
            Self (for method chaining)

        Example:
            >>> config1.merge(config2)
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot merge {other.__class__.__name__} into {self.__class__.__name__}"
            )

        other_dict = other.to_dict()
        self.update(**other_dict)
        return self

    def __repr__(self) -> str:
        """String representation of configuration."""
        params = ", ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({params})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"{self.__class__.__name__}:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __eq__(self, other) -> bool:
        """Check equality with another configuration."""
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()
