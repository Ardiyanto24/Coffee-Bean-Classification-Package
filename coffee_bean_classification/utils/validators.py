"""Validation utilities for configurations and inputs."""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


def validate_path(path: Union[str, Path], must_exist: bool = True, path_type: str = "any") -> bool:
    """
    Validate file or directory path.

    Args:
        path: Path to validate
        must_exist: If True, path must exist
        path_type: Type of path ('file', 'dir', 'any')

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_path('/path/to/dataset', must_exist=True, path_type='dir')
    """
    path = Path(path)

    # Check existence
    if must_exist and not path.exists():
        raise ValidationError(f"Path does not exist: {path}")

    # Check type
    if path.exists():
        if path_type == "file" and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        elif path_type == "dir" and not path.is_dir():
            raise ValidationError(f"Path is not a directory: {path}")

    logger.debug(f"Path validated: {path}")
    return True


def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    inclusive: bool = True,
) -> bool:
    """
    Validate that value is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        inclusive: If True, endpoints are included

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_range(0.5, min_val=0.0, max_val=1.0)
    """
    if min_val is not None:
        if inclusive and value < min_val:
            raise ValidationError(f"Value {value} < minimum {min_val}")
        elif not inclusive and value <= min_val:
            raise ValidationError(f"Value {value} <= minimum {min_val}")

    if max_val is not None:
        if inclusive and value > max_val:
            raise ValidationError(f"Value {value} > maximum {max_val}")
        elif not inclusive and value >= max_val:
            raise ValidationError(f"Value {value} >= maximum {max_val}")

    logger.debug(f"Range validated: {value}")
    return True


def validate_type(value: Any, expected_type: type, allow_none: bool = False) -> bool:
    """
    Validate value type.

    Args:
        value: Value to validate
        expected_type: Expected type
        allow_none: If True, None is allowed

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if allow_none:
            return True
        else:
            raise ValidationError("Value cannot be None")

    if not isinstance(value, expected_type):
        raise ValidationError(f"Expected type {expected_type.__name__}, got {type(value).__name__}")

    logger.debug(f"Type validated: {type(value).__name__}")
    return True


def validate_choice(value: Any, choices: List[Any]) -> bool:
    """
    Validate that value is in list of choices.

    Args:
        value: Value to validate
        choices: List of allowed values

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_choice('adam', ['adam', 'sgd', 'rmsprop'])
    """
    if value not in choices:
        raise ValidationError(f"Invalid choice: {value}. Must be one of {choices}")

    logger.debug(f"Choice validated: {value}")
    return True


def validate_positive(value: Union[int, float], strict: bool = True) -> bool:
    """
    Validate that value is positive.

    Args:
        value: Value to validate
        strict: If True, value must be > 0. If False, value must be >= 0

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if strict and value <= 0:
        raise ValidationError(f"Value must be positive (> 0), got {value}")
    elif not strict and value < 0:
        raise ValidationError(f"Value must be non-negative (>= 0), got {value}")

    logger.debug(f"Positive value validated: {value}")
    return True


def validate_shape(
    array: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None, ndim: Optional[int] = None
) -> bool:
    """
    Validate array shape.

    Args:
        array: Array to validate
        expected_shape: Expected shape (None for any dimension)
        ndim: Expected number of dimensions

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_shape(image, expected_shape=(224, 224, 3))
        >>> validate_shape(batch, ndim=4)  # Batch of images
    """
    if ndim is not None and array.ndim != ndim:
        raise ValidationError(f"Expected {ndim} dimensions, got {array.ndim}")

    if expected_shape is not None:
        if len(expected_shape) != array.ndim:
            raise ValidationError(
                f"Shape mismatch: expected {len(expected_shape)} dims, " f"got {array.ndim} dims"
            )

        for i, (expected, actual) in enumerate(zip(expected_shape, array.shape)):
            if expected is not None and expected != actual:
                raise ValidationError(f"Dimension {i}: expected {expected}, got {actual}")

    logger.debug(f"Shape validated: {array.shape}")
    return True


def validate_ratio(ratio: Union[float, Tuple[float, ...]], total: float = 1.0) -> bool:
    """
    Validate that ratio(s) sum to expected total.

    Args:
        ratio: Single ratio or tuple of ratios
        total: Expected sum

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_ratio((0.7, 0.2, 0.1))  # Train/val/test split
    """
    if isinstance(ratio, (tuple, list)):
        ratio_sum = sum(ratio)
        if not np.isclose(ratio_sum, total):
            raise ValidationError(f"Ratios sum to {ratio_sum}, expected {total}")

        # Check all ratios are positive
        for r in ratio:
            if r < 0:
                raise ValidationError(f"Ratio cannot be negative: {r}")
    else:
        if ratio < 0 or ratio > total:
            raise ValidationError(f"Ratio {ratio} not in valid range [0, {total}]")

    logger.debug(f"Ratio validated: {ratio}")
    return True


def validate_config_dict(
    config: dict, required_keys: List[str], optional_keys: Optional[List[str]] = None
) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        optional_keys: List of optional keys

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_config_dict(
        ...     config,
        ...     required_keys=['epochs', 'batch_size'],
        ...     optional_keys=['learning_rate']
        ... )
    """
    # Check required keys
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ValidationError(f"Missing required keys: {missing_keys}")

    # Check for unknown keys
    if optional_keys is not None:
        allowed_keys = set(required_keys) | set(optional_keys)
        unknown_keys = set(config.keys()) - allowed_keys
        if unknown_keys:
            logger.warning(f"Unknown config keys: {unknown_keys}")

    logger.debug("Config dictionary validated")
    return True


def validate_image_size(size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Validate and normalize image size.

    Args:
        size: Single int or (height, width) tuple

    Returns:
        Normalized (height, width) tuple

    Raises:
        ValidationError: If validation fails

    Example:
        >>> size = validate_image_size(224)  # Returns (224, 224)
        >>> size = validate_image_size((224, 224))  # Returns (224, 224)
    """
    if isinstance(size, int):
        validate_positive(size)
        return (size, size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        validate_positive(size[0])
        validate_positive(size[1])
        return tuple(size)
    else:
        raise ValidationError(f"Image size must be int or (height, width) tuple, got {size}")


class ConfigValidator:
    """
    Reusable validator for configuration objects.

    Example:
        >>> validator = ConfigValidator()
        >>> validator.add_rule('epochs', validate_positive)
        >>> validator.add_rule('batch_size', lambda x: validate_range(x, 1, 512))
        >>> validator.validate({'epochs': 50, 'batch_size': 32})
    """

    def __init__(self):
        """Initialize validator."""
        self.rules = {}

    def add_rule(self, key: str, validation_func):
        """
        Add validation rule for a key.

        Args:
            key: Config key to validate
            validation_func: Function that validates the value
        """
        self.rules[key] = validation_func
        logger.debug(f"Added validation rule for: {key}")

    def validate(self, config: dict) -> bool:
        """
        Validate configuration dictionary.

        Args:
            config: Configuration to validate

        Returns:
            True if all validations pass

        Raises:
            ValidationError: If any validation fails
        """
        for key, validation_func in self.rules.items():
            if key in config:
                try:
                    validation_func(config[key])
                except ValidationError as e:
                    raise ValidationError(f"Validation failed for '{key}': {str(e)}")

        logger.info("All validations passed")
        return True
