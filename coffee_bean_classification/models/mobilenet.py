"""MobileNetV3 model implementation."""

import tensorflow as tf
from typing import Optional

from .base import BaseModel
from ..utils import get_logger

logger = get_logger(__name__)


class MobileNetV3Model(BaseModel):
    """
    MobileNetV3Small model for image classification.

    MobileNetV3 is designed for mobile and edge devices with efficient
    architecture using depthwise separable convolutions.

    Paper: "Searching for MobileNetV3"
    https://arxiv.org/abs/1905.02244

    Optimal input size: 224x224

    Features:
    - Lightweight architecture
    - Fast inference
    - Good accuracy-efficiency trade-off

    Example:
        >>> config = ModelConfig.for_architecture('mobilenet_v3', num_classes=4)
        >>> model = MobileNetV3Model(config)
        >>> keras_model = model.build()
    """

    def __init__(self, config):
        """Initialize MobileNetV3 model."""
        super().__init__(config)
        self.backbone = None
        logger.info("Initialized MobileNetV3Model")

    def build(self) -> tf.keras.Model:
        """Build MobileNetV3Small model architecture."""
        logger.info("Building MobileNetV3Small model...")

        # Input layer
        inputs = tf.keras.Input(shape=self.config.input_shape, name="input")

        # Load MobileNetV3Small backbone
        self.backbone = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights=self.config.weights,
            input_shape=self.config.input_shape,
            pooling=self.config.pooling,
            minimalistic=False,  # Use full architecture
        )

        # Freeze backbone if specified
        if self.config.freeze_backbone:
            self.backbone.trainable = False
            logger.info("Backbone frozen for transfer learning")

        # Forward pass
        x = self.backbone(inputs, training=False)

        # Classification head
        x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate, name="dropout_head")(x)

        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config.num_classes, activation=self.config.activation, name="predictions"
        )(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV3Small")

        params = self.count_parameters()
        logger.info(
            f"MobileNetV3Small built: {params['total']:,} total params "
            f"({params['trainable']:,} trainable)"
        )

        return self.model

    def freeze_backbone(self) -> None:
        """Freeze the MobileNetV3 backbone."""
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return
        self.backbone.trainable = False
        logger.info("MobileNetV3 backbone frozen")

    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        """Unfreeze the MobileNetV3 backbone."""
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return

        if from_layer is not None:
            for layer in self.backbone.layers[:from_layer]:
                layer.trainable = False
            for layer in self.backbone.layers[from_layer:]:
                layer.trainable = True
            logger.info(f"MobileNetV3 backbone unfrozen from layer {from_layer}")
        else:
            self.backbone.trainable = True
            logger.info("MobileNetV3 backbone fully unfrozen")

    def get_model_size_mb(self) -> float:
        """
        Estimate model size in MB.

        Returns:
            Approximate model size in megabytes
        """
        if self.model is None:
            logger.warning("Model not built yet")
            return 0.0

        # Estimate: 4 bytes per parameter (float32)
        params = self.count_parameters()
        size_mb = (params["total"] * 4) / (1024 * 1024)

        logger.info(f"Estimated model size: {size_mb:.2f} MB")
        return size_mb
