"""ResNet50 model implementation."""

from typing import Optional

import tensorflow as tf

from ..utils import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class ResNet50Model(BaseModel):
    """
    ResNet50 model for image classification.

    ResNet (Residual Network) uses skip connections to enable training
    of very deep networks. ResNet50 has 50 layers.

    Paper: "Deep Residual Learning for Image Recognition"
    https://arxiv.org/abs/1512.03385

    Attributes:
        config: Model configuration
        backbone: ResNet50 backbone

    Example:
        >>> from coffee_bean_classification.configs import ModelConfig
        >>> config = ModelConfig.for_architecture('resnet50', num_classes=4)
        >>> model = ResNet50Model(config)
        >>> keras_model = model.build()
    """

    def __init__(self, config):
        """
        Initialize ResNet50 model.

        Args:
            config: ModelConfig instance
        """
        super().__init__(config)
        self.backbone = None
        logger.info("Initialized ResNet50Model")

    def build(self) -> tf.keras.Model:
        """
        Build ResNet50 model architecture.

        Returns:
            Compiled Keras model
        """
        logger.info("Building ResNet50 model...")

        # Input layer
        inputs = tf.keras.Input(shape=self.config.input_shape, name="input")

        # Load ResNet50 backbone
        self.backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights=self.config.weights,
            input_shape=self.config.input_shape,
            pooling=self.config.pooling,
        )

        # Freeze backbone if specified
        if self.config.freeze_backbone:
            self.backbone.trainable = False
            logger.info("Backbone frozen for transfer learning")
        else:
            logger.info("Backbone trainable")

        # Forward pass through backbone
        x = self.backbone(inputs, training=False)

        # Classification head
        x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate, name="dropout_head")(x)

        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config.num_classes, activation=self.config.activation, name="predictions"
        )(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50")

        # Log model info
        params = self.count_parameters()
        logger.info(
            f"ResNet50 built: {params['total']:,} total params "
            f"({params['trainable']:,} trainable)"
        )

        return self.model

    def freeze_backbone(self) -> None:
        """Freeze the ResNet50 backbone for transfer learning."""
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return

        self.backbone.trainable = False
        logger.info("ResNet50 backbone frozen")

    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        """
        Unfreeze the ResNet50 backbone for fine-tuning.

        Args:
            from_layer: If specified, only unfreeze layers from this index onwards
        """
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return

        if from_layer is not None:
            # Unfreeze from specific layer
            for layer in self.backbone.layers[:from_layer]:
                layer.trainable = False
            for layer in self.backbone.layers[from_layer:]:
                layer.trainable = True
            logger.info(f"ResNet50 backbone unfrozen from layer {from_layer}")
        else:
            # Unfreeze all
            self.backbone.trainable = True
            logger.info("ResNet50 backbone fully unfrozen")

    def get_feature_extractor(self) -> tf.keras.Model:
        """
        Get feature extractor (without classification head).

        Returns:
            Keras model that outputs features
        """
        if self.model is None:
            logger.warning("Model not built yet")
            return None

        feature_extractor = tf.keras.Model(
            inputs=self.model.input, outputs=self.backbone.output, name="ResNet50_features"
        )

        logger.info("Created ResNet50 feature extractor")
        return feature_extractor
