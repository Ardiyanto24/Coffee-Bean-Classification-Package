"""Model factory for creating different model architectures."""

from typing import Dict, List, Optional, Type

import tensorflow as tf

from ..utils import ValidationError, get_logger
from .base import BaseModel
from .densenet import DenseNet121Model
from .efficientnet import EfficientNetB0Model, EfficientNetB3Model
from .mobilenet import MobileNetV3Model
from .resnet import ResNet50Model

logger = get_logger(__name__)


class ModelFactory:
    """
    Factory for creating model instances.

    Uses registry pattern to allow easy addition of new models.

    Supported architectures:
    - resnet50: ResNet50
    - efficientnet_b0: EfficientNetB0
    - efficientnet_b3: EfficientNetB3
    - mobilenet_v3: MobileNetV3Small
    - densenet121: DenseNet121

    Example:
        >>> from coffee_bean_classification.configs import ModelConfig
        >>> config = ModelConfig.for_architecture('resnet50', num_classes=4)
        >>> model = ModelFactory.create('resnet50', config)
        >>> keras_model = model.build()

    Custom model registration:
        >>> @ModelFactory.register('my_custom_model')
        >>> class MyCustomModel(BaseModel):
        >>>     def build(self):
        >>>         # implementation
        >>>         pass
    """

    # Registry of available models
    _registry: Dict[str, Type[BaseModel]] = {
        "resnet50": ResNet50Model,
        "efficientnet_b0": EfficientNetB0Model,
        "efficientnet_b3": EfficientNetB3Model,
        "mobilenet_v3": MobileNetV3Model,
        "densenet121": DenseNet121Model,
    }

    @classmethod
    def create(cls, model_name: str, config) -> BaseModel:
        """
        Create a model instance.

        Args:
            model_name: Name of the model architecture
            config: ModelConfig instance

        Returns:
            Model instance (subclass of BaseModel)

        Raises:
            ValidationError: If model_name is not registered

        Example:
            >>> config = ModelConfig(architecture='resnet50', num_classes=4)
            >>> model = ModelFactory.create('resnet50', config)
        """
        model_name = model_name.lower()

        if model_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValidationError(
                f"Model '{model_name}' not found in registry. " f"Available models: {available}"
            )

        model_class = cls._registry[model_name]
        logger.info(f"Creating model: {model_name}")

        return model_class(config)

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a custom model.

        Args:
            name: Name to register the model under

        Returns:
            Decorator function

        Example:
            >>> @ModelFactory.register('my_model')
            >>> class MyModel(BaseModel):
            >>>     def build(self):
            >>>         # implementation
            >>>         pass
        """

        def decorator(model_class: Type[BaseModel]):
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"{model_class.__name__} must be a subclass of BaseModel")

            cls._registry[name.lower()] = model_class
            logger.info(f"Registered custom model: {name}")
            return model_class

        return decorator

    @classmethod
    def list_available(cls) -> List[str]:
        """
        Get list of available model architectures.

        Returns:
            List of registered model names

        Example:
            >>> available = ModelFactory.list_available()
            >>> print(available)
            ['resnet50', 'efficientnet_b0', 'efficientnet_b3',
             'mobilenet_v3', 'densenet121']
        """
        return sorted(cls._registry.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, any]:
        """
        Get information about a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information

        Example:
            >>> info = ModelFactory.get_model_info('resnet50')
            >>> print(info['class_name'])
            'ResNet50Model'
        """
        model_name = model_name.lower()

        if model_name not in cls._registry:
            raise ValidationError(f"Model '{model_name}' not found")

        model_class = cls._registry[model_name]

        return {
            "name": model_name,
            "class_name": model_class.__name__,
            "module": model_class.__module__,
            "docstring": model_class.__doc__,
        }

    @classmethod
    def create_all(cls, config) -> Dict[str, BaseModel]:
        """
        Create instances of all registered models.

        Args:
            config: ModelConfig instance (will be used for all models)

        Returns:
            Dictionary mapping model names to instances

        Example:
            >>> config = ModelConfig(num_classes=4)
            >>> models = ModelFactory.create_all(config)
            >>> for name, model in models.items():
            >>>     print(f"{name}: {model}")
        """
        models = {}

        for model_name in cls._registry.keys():
            try:
                models[model_name] = cls.create(model_name, config)
            except Exception as e:
                logger.error(f"Failed to create {model_name}: {e}")

        logger.info(f"Created {len(models)} models")
        return models

    @classmethod
    def build_all(cls, config) -> Dict[str, tf.keras.Model]:
        """
        Create and build all registered models.

        Args:
            config: ModelConfig instance

        Returns:
            Dictionary mapping model names to built Keras models

        Example:
            >>> config = ModelConfig(num_classes=4)
            >>> keras_models = ModelFactory.build_all(config)
        """
        models = cls.create_all(config)
        keras_models = {}

        for name, model in models.items():
            try:
                keras_models[name] = model.build()
                logger.info(f"✓ Built {name}")
            except Exception as e:
                logger.error(f"✗ Failed to build {name}: {e}")

        return keras_models

    @classmethod
    def compare_models(cls) -> Dict[str, Dict]:
        """
        Compare all registered models (approximate sizes and complexity).

        Returns:
            Dictionary with model comparison information

        Example:
            >>> comparison = ModelFactory.compare_models()
            >>> for name, info in comparison.items():
            >>>     print(f"{name}: {info['approximate_params']} parameters")
        """
        # Approximate parameter counts (from literature)
        # These are rough estimates without building the models
        estimates = {
            "resnet50": {
                "approximate_params": 25_600_000,
                "optimal_input_size": (224, 224),
                "relative_speed": "medium",
                "relative_accuracy": "high",
            },
            "efficientnet_b0": {
                "approximate_params": 5_300_000,
                "optimal_input_size": (224, 224),
                "relative_speed": "fast",
                "relative_accuracy": "high",
            },
            "efficientnet_b3": {
                "approximate_params": 12_000_000,
                "optimal_input_size": (300, 300),
                "relative_speed": "medium",
                "relative_accuracy": "very_high",
            },
            "mobilenet_v3": {
                "approximate_params": 2_500_000,
                "optimal_input_size": (224, 224),
                "relative_speed": "very_fast",
                "relative_accuracy": "medium",
            },
            "densenet121": {
                "approximate_params": 8_000_000,
                "optimal_input_size": (224, 224),
                "relative_speed": "slow",
                "relative_accuracy": "high",
            },
        }

        comparison = {}
        for model_name in cls._registry.keys():
            if model_name in estimates:
                comparison[model_name] = estimates[model_name]
            else:
                comparison[model_name] = {
                    "approximate_params": "unknown",
                    "optimal_input_size": "unknown",
                }

        return comparison

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            model_name: Name of the model

        Returns:
            True if registered, False otherwise
        """
        return model_name.lower() in cls._registry

    @classmethod
    def unregister(cls, model_name: str) -> bool:
        """
        Unregister a model.

        Args:
            model_name: Name of the model to unregister

        Returns:
            True if unregistered, False if not found
        """
        model_name = model_name.lower()

        if model_name in cls._registry:
            del cls._registry[model_name]
            logger.info(f"Unregistered model: {model_name}")
            return True

        logger.warning(f"Model '{model_name}' not found in registry")
        return False

    @classmethod
    def get_registry(cls) -> Dict[str, Type[BaseModel]]:
        """
        Get the full model registry.

        Returns:
            Dictionary mapping model names to model classes
        """
        return cls._registry.copy()
