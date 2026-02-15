"""EfficientNet model implementations."""

import tensorflow as tf
from typing import Optional

from .base import BaseModel
from ..utils import get_logger

logger = get_logger(__name__)


class EfficientNetB0Model(BaseModel):
    """
    EfficientNetB0 model for image classification.
    
    EfficientNet uses compound scaling to balance network depth, width,
    and resolution. B0 is the base model.
    
    Paper: "EfficientNet: Rethinking Model Scaling for CNNs"
    https://arxiv.org/abs/1905.11946
    
    Optimal input size: 224x224
    
    Example:
        >>> config = ModelConfig.for_architecture('efficientnet_b0', num_classes=4)
        >>> model = EfficientNetB0Model(config)
        >>> keras_model = model.build()
    """
    
    def __init__(self, config):
        """Initialize EfficientNetB0 model."""
        super().__init__(config)
        self.backbone = None
        logger.info("Initialized EfficientNetB0Model")
    
    def build(self) -> tf.keras.Model:
        """Build EfficientNetB0 model architecture."""
        logger.info("Building EfficientNetB0 model...")
        
        # Input layer
        inputs = tf.keras.Input(shape=self.config.input_shape, name='input')
        
        # Load EfficientNetB0 backbone
        self.backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=self.config.weights,
            input_shape=self.config.input_shape,
            pooling=self.config.pooling
        )
        
        # Freeze backbone if specified
        if self.config.freeze_backbone:
            self.backbone.trainable = False
            logger.info("Backbone frozen for transfer learning")
        
        # Forward pass
        x = self.backbone(inputs, training=False)
        
        # Classification head
        x = tf.keras.layers.BatchNormalization(name='bn_head')(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate, name='dropout_head')(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config.num_classes,
            activation=self.config.activation,
            name='predictions'
        )(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='EfficientNetB0')
        
        params = self.count_parameters()
        logger.info(
            f"EfficientNetB0 built: {params['total']:,} total params "
            f"({params['trainable']:,} trainable)"
        )
        
        return self.model
    
    def freeze_backbone(self) -> None:
        """Freeze the EfficientNetB0 backbone."""
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return
        self.backbone.trainable = False
        logger.info("EfficientNetB0 backbone frozen")
    
    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        """Unfreeze the EfficientNetB0 backbone."""
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return
        
        if from_layer is not None:
            for layer in self.backbone.layers[:from_layer]:
                layer.trainable = False
            for layer in self.backbone.layers[from_layer:]:
                layer.trainable = True
            logger.info(f"EfficientNetB0 backbone unfrozen from layer {from_layer}")
        else:
            self.backbone.trainable = True
            logger.info("EfficientNetB0 backbone fully unfrozen")


class EfficientNetB3Model(BaseModel):
    """
    EfficientNetB3 model for image classification.
    
    Larger version of EfficientNet with better accuracy but more parameters.
    
    Optimal input size: 300x300
    
    Example:
        >>> config = ModelConfig.for_architecture('efficientnet_b3', num_classes=4)
        >>> model = EfficientNetB3Model(config)
        >>> keras_model = model.build()
    """
    
    def __init__(self, config):
        """Initialize EfficientNetB3 model."""
        super().__init__(config)
        self.backbone = None
        logger.info("Initialized EfficientNetB3Model")
    
    def build(self) -> tf.keras.Model:
        """Build EfficientNetB3 model architecture."""
        logger.info("Building EfficientNetB3 model...")
        
        # Input layer
        inputs = tf.keras.Input(shape=self.config.input_shape, name='input')
        
        # Load EfficientNetB3 backbone
        self.backbone = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights=self.config.weights,
            input_shape=self.config.input_shape,
            pooling=self.config.pooling
        )
        
        # Freeze backbone if specified
        if self.config.freeze_backbone:
            self.backbone.trainable = False
            logger.info("Backbone frozen for transfer learning")
        
        # Forward pass
        x = self.backbone(inputs, training=False)
        
        # Classification head
        x = tf.keras.layers.BatchNormalization(name='bn_head')(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate, name='dropout_head')(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config.num_classes,
            activation=self.config.activation,
            name='predictions'
        )(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='EfficientNetB3')
        
        params = self.count_parameters()
        logger.info(
            f"EfficientNetB3 built: {params['total']:,} total params "
            f"({params['trainable']:,} trainable)"
        )
        
        return self.model
    
    def freeze_backbone(self) -> None:
        """Freeze the EfficientNetB3 backbone."""
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return
        self.backbone.trainable = False
        logger.info("EfficientNetB3 backbone frozen")
    
    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        """Unfreeze the EfficientNetB3 backbone."""
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return
        
        if from_layer is not None:
            for layer in self.backbone.layers[:from_layer]:
                layer.trainable = False
            for layer in self.backbone.layers[from_layer:]:
                layer.trainable = True
            logger.info(f"EfficientNetB3 backbone unfrozen from layer {from_layer}")
        else:
            self.backbone.trainable = True
            logger.info("EfficientNetB3 backbone fully unfrozen")
