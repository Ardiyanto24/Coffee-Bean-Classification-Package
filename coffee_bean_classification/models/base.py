"""Base model class for all neural network models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import tensorflow as tf

from ..utils import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all model architectures.

    All model classes must inherit from this base class and implement
    the build() method.

    Attributes:
        config: Model configuration
        model: Compiled Keras model

    Example:
        >>> class MyModel(BaseModel):
        ...     def build(self) -> tf.keras.Model:
        ...         # Build model architecture
        ...         return model
    """

    def __init__(self, config):
        """
        Initialize base model.

        Args:
            config: Model configuration object
        """
        self.config = config
        self.model: Optional[tf.keras.Model] = None
        self._compiled = False

        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """
        Build the model architecture.

        This method must be implemented by all subclasses to define
        their specific model architecture.

        Returns:
            Keras model
        """
        pass

    def compile(
        self,
        optimizer: Optional[Any] = None,
        loss: Optional[Any] = None,
        metrics: Optional[List[Any]] = None,
        **kwargs,
    ) -> None:
        """
        Compile the model.

        Args:
            optimizer: Keras optimizer or string
            loss: Loss function
            metrics: List of metrics
            **kwargs: Additional arguments for model.compile()
        """
        if self.model is None:
            logger.info("Model not built yet, building now...")
            self.model = self.build()

        # Use config defaults if not provided
        if optimizer is None:
            optimizer = getattr(self.config, "optimizer", "adam")
        if loss is None:
            loss = getattr(self.config, "loss", "categorical_crossentropy")
        if metrics is None:
            metrics = getattr(self.config, "metrics", ["accuracy"])

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

        self._compiled = True
        logger.info(f"{self.__class__.__name__} compiled successfully")

    def summary(self, **kwargs) -> None:
        """
        Print model summary.

        Args:
            **kwargs: Arguments for model.summary()
        """
        if self.model is None:
            logger.warning("Model not built yet")
            return

        self.model.summary(**kwargs)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.

        Returns:
            Dictionary with parameter counts
        """
        if self.model is None:
            logger.warning("Model not built yet")
            return {"total": 0, "trainable": 0, "non_trainable": 0}

        trainable = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])

        return {
            "total": int(trainable + non_trainable),
            "trainable": int(trainable),
            "non_trainable": int(non_trainable),
        }

    def get_model(self) -> tf.keras.Model:
        """
        Get the Keras model.

        Returns:
            Keras model instance
        """
        if self.model is None:
            logger.info("Building model...")
            self.model = self.build()
        return self.model

    def save(self, filepath: str, **kwargs) -> None:
        """
        Save model to file.

        Args:
            filepath: Path to save model
            **kwargs: Additional arguments for model.save()
        """
        if self.model is None:
            raise ValueError("No model to save. Build the model first.")

        self.model.save(filepath, **kwargs)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, config=None, **kwargs) -> "BaseModel":
        """
        Load model from file.

        Args:
            filepath: Path to model file
            config: Model configuration
            **kwargs: Additional arguments for tf.keras.models.load_model()

        Returns:
            Model instance with loaded weights
        """
        loaded_model = tf.keras.models.load_model(filepath, **kwargs)

        if config is None:
            # Create a minimal config
            from ..configs.model import ModelConfig

            config = ModelConfig(
                architecture=cls.__name__,
                input_shape=loaded_model.input_shape[1:],
                num_classes=loaded_model.output_shape[-1],
            )

        instance = cls(config)
        instance.model = loaded_model
        instance._compiled = True

        logger.info(f"Model loaded from {filepath}")
        return instance

    def freeze_backbone(self) -> None:
        """Freeze the backbone layers for transfer learning."""
        if self.model is None:
            raise ValueError("Model not built yet")

        # This should be implemented by subclasses for specific architectures
        logger.warning("freeze_backbone() should be implemented by subclass")

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone layers for fine-tuning."""
        if self.model is None:
            raise ValueError("Model not built yet")

        # This should be implemented by subclasses for specific architectures
        logger.warning("unfreeze_backbone() should be implemented by subclass")

    def __repr__(self) -> str:
        """String representation."""
        param_count = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"params={param_count['total']:,}, "
            f"trainable={param_count['trainable']:,})"
        )
