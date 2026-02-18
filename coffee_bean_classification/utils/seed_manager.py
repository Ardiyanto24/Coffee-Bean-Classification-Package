"""Seed management for reproducibility."""

import os
import random
import numpy as np
import tensorflow as tf

from .logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global random seed for reproducibility across all libraries.

    This function sets seeds for:
    - Python's random module
    - NumPy
    - TensorFlow
    - Environment variables

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic operations (may impact performance)

    Example:
        >>> set_global_seed(42)
        >>> # All random operations will now be reproducible
    """
    logger.info(f"Setting global seed to {seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # TensorFlow
    tf.random.set_seed(seed)

    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow deterministic operations
    if deterministic:
        # Enable deterministic operations
        tf.config.experimental.enable_op_determinism()
        logger.info("Enabled deterministic operations (may reduce performance)")

    # Additional TensorFlow configurations for reproducibility
    try:
        # Set TF random seed using new API
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        # Fallback for older TensorFlow versions
        pass

    logger.debug(f"Seed {seed} set for: random, numpy, tensorflow")


class SeedContext:
    """
    Context manager for temporary seed setting.

    Example:
        >>> with SeedContext(42):
        >>>     # Operations here are deterministic
        >>>     data = np.random.rand(10)
        >>> # Original seed restored after context
    """

    def __init__(self, seed: int, deterministic: bool = True):
        """
        Initialize seed context.

        Args:
            seed: Temporary seed value
            deterministic: Enable deterministic operations
        """
        self.seed = seed
        self.deterministic = deterministic

        # Store current states
        self.python_state = None
        self.numpy_state = None
        self.tf_seed = None

    def __enter__(self):
        """Save current states and set new seed."""
        # Save current states
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()

        # Set new seed
        set_global_seed(self.seed, self.deterministic)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original states."""
        # Restore states
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)


class SeedManager:
    """
    Manages seeds for different components of the training pipeline.

    Attributes:
        global_seed: Main seed for the entire experiment
        data_seed: Seed for data operations
        model_seed: Seed for model initialization
        augmentation_seed: Seed for data augmentation
    """

    def __init__(self, global_seed: int = 42):
        """
        Initialize seed manager.

        Args:
            global_seed: Main seed value
        """
        self.global_seed = global_seed

        # Derive component-specific seeds from global seed
        # This ensures reproducibility while allowing some variation
        self.data_seed = global_seed
        self.model_seed = global_seed + 1
        self.augmentation_seed = global_seed + 2
        self.validation_seed = global_seed + 3

        logger.info(f"SeedManager initialized with global_seed={global_seed}")

    def set_all_seeds(self, deterministic: bool = True) -> None:
        """
        Set all seeds at once.

        Args:
            deterministic: Enable deterministic operations
        """
        set_global_seed(self.global_seed, deterministic)

    def get_data_seed(self) -> int:
        """Get seed for data operations."""
        return self.data_seed

    def get_model_seed(self) -> int:
        """Get seed for model initialization."""
        return self.model_seed

    def get_augmentation_seed(self) -> int:
        """Get seed for augmentation."""
        return self.augmentation_seed

    def get_validation_seed(self) -> int:
        """Get seed for validation split."""
        return self.validation_seed

    def to_dict(self) -> dict:
        """Export seeds as dictionary."""
        return {
            "global_seed": self.global_seed,
            "data_seed": self.data_seed,
            "model_seed": self.model_seed,
            "augmentation_seed": self.augmentation_seed,
            "validation_seed": self.validation_seed,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "SeedManager":
        """
        Create SeedManager from dictionary.

        Args:
            config: Dictionary with seed configuration

        Returns:
            SeedManager instance
        """
        manager = cls(config["global_seed"])
        manager.data_seed = config.get("data_seed", manager.data_seed)
        manager.model_seed = config.get("model_seed", manager.model_seed)
        manager.augmentation_seed = config.get("augmentation_seed", manager.augmentation_seed)
        manager.validation_seed = config.get("validation_seed", manager.validation_seed)
        return manager


def verify_reproducibility(seed: int, n_trials: int = 3) -> bool:
    """
    Verify that operations are reproducible with the given seed.

    Args:
        seed: Seed to test
        n_trials: Number of trials to run

    Returns:
        True if reproducible, False otherwise

    Example:
        >>> is_reproducible = verify_reproducibility(42, n_trials=5)
    """
    logger.info(f"Verifying reproducibility with seed={seed}, n_trials={n_trials}")

    results = []
    for i in range(n_trials):
        set_global_seed(seed)

        # Test operations
        rand_val = random.random()
        np_val = np.random.rand()
        tf_val = float(tf.random.normal([1]).numpy()[0])

        results.append((rand_val, np_val, tf_val))

    # Check if all trials produced same results
    first_result = results[0]
    is_reproducible = all(r == first_result for r in results)

    if is_reproducible:
        logger.info("✓ Reproducibility verified")
    else:
        logger.warning("✗ Results are not reproducible")
        logger.debug(f"Results: {results}")

    return is_reproducible
