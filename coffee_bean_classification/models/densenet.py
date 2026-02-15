"""DenseNet121 model implementation."""

import tensorflow as tf
from typing import Optional

from .base import BaseModel
from ..utils import get_logger

logger = get_logger(__name__)


class DenseNet121Model(BaseModel):
    """
    DenseNet121 model for image classification.
    
    DenseNet (Densely Connected Network) connects each layer to every other
    layer in a feed-forward fashion, promoting feature reuse and reducing
    the number of parameters.
    
    Paper: "Densely Connected Convolutional Networks"
    https://arxiv.org/abs/1608.06993
    
    Optimal input size: 224x224
    
    Features:
    - Dense connectivity pattern
    - Efficient parameter usage
    - Strong gradient flow
    - Feature reuse
    
    Example:
        >>> config = ModelConfig.for_architecture('densenet121', num_classes=4)
        >>> model = DenseNet121Model(config)
        >>> keras_model = model.build()
    """
    
    def __init__(self, config):
        """Initialize DenseNet121 model."""
        super().__init__(config)
        self.backbone = None
        logger.info("Initialized DenseNet121Model")
    
    def build(self) -> tf.keras.Model:
        """Build DenseNet121 model architecture."""
        logger.info("Building DenseNet121 model...")
        
        # Input layer
        inputs = tf.keras.Input(shape=self.config.input_shape, name='input')
        
        # Load DenseNet121 backbone
        self.backbone = tf.keras.applications.DenseNet121(
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
        
        # Additional dense layer for better feature learning
        x = tf.keras.layers.Dense(256, activation='relu', name='dense_intermediate')(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate * 0.5, name='dropout_intermediate')(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config.num_classes,
            activation=self.config.activation,
            name='predictions'
        )(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='DenseNet121')
        
        params = self.count_parameters()
        logger.info(
            f"DenseNet121 built: {params['total']:,} total params "
            f"({params['trainable']:,} trainable)"
        )
        
        return self.model
    
    def freeze_backbone(self) -> None:
        """Freeze the DenseNet121 backbone."""
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return
        self.backbone.trainable = False
        logger.info("DenseNet121 backbone frozen")
    
    def unfreeze_backbone(self, from_layer: Optional[int] = None) -> None:
        """
        Unfreeze the DenseNet121 backbone.
        
        Args:
            from_layer: If specified, unfreeze from this layer onwards
        """
        if self.backbone is None:
            logger.warning("Backbone not initialized yet")
            return
        
        if from_layer is not None:
            for layer in self.backbone.layers[:from_layer]:
                layer.trainable = False
            for layer in self.backbone.layers[from_layer:]:
                layer.trainable = True
            logger.info(f"DenseNet121 backbone unfrozen from layer {from_layer}")
        else:
            self.backbone.trainable = True
            logger.info("DenseNet121 backbone fully unfrozen")
    
    def freeze_bn_layers(self) -> None:
        """
        Freeze all BatchNormalization layers.
        
        This is useful for fine-tuning to maintain the statistics
        learned during pre-training.
        """
        if self.model is None:
            logger.warning("Model not built yet")
            return
        
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        
        logger.info("All BatchNormalization layers frozen")
    
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
            inputs=self.model.input,
            outputs=self.backbone.output,
            name='DenseNet121_features'
        )
        
        logger.info("Created DenseNet121 feature extractor")
        return feature_extractor
