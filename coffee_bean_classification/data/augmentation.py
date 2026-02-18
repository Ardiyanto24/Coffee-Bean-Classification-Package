"""Data augmentation strategies for image classification."""

import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Any

from ..utils import get_logger

logger = get_logger(__name__)


class DataAugmentation:
    """
    Data augmentation for image classification.
    
    Provides various augmentation strategies with configurable parameters.
    
    Attributes:
        config: Augmentation configuration dictionary
        strategy: Augmentation strategy name
        
    Example:
        >>> augmentation = DataAugmentation(strategy='medium')
        >>> augmented_image = augmentation.apply(image)
    """
    
    # Predefined augmentation strategies
    STRATEGIES = {
        'none': {},
        'light': {
            'horizontal_flip': True,
            'rotation_range': 0.1,
            'zoom_range': 0.05,
        },
        'medium': {
            'horizontal_flip': True,
            'rotation_range': 0.2,
            'zoom_range': 0.1,
            'brightness_range': (0.8, 1.2),
        },
        'heavy': {
            'horizontal_flip': True,
            'vertical_flip': True,
            'rotation_range': 0.3,
            'zoom_range': 0.15,
            'brightness_range': (0.7, 1.3),
            'contrast_range': (0.7, 1.3),
        }
    }
    
    def __init__(
        self,
        strategy: str = 'medium',
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize data augmentation.
        
        Args:
            strategy: Predefined strategy name ('none', 'light', 'medium', 'heavy')
            custom_config: Custom augmentation configuration (overrides strategy)
        """
        if custom_config is not None:
            self.config = custom_config
            self.strategy = 'custom'
        else:
            if strategy not in self.STRATEGIES:
                logger.warning(
                    f"Unknown strategy '{strategy}', using 'medium'. "
                    f"Available: {list(self.STRATEGIES.keys())}"
                )
                strategy = 'medium'
            self.config = self.STRATEGIES[strategy].copy()
            self.strategy = strategy
        
        logger.info(f"DataAugmentation initialized with strategy: {self.strategy}")
        logger.debug(f"Augmentation config: {self.config}")
    
    def apply(self, image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Apply augmentation to image.
        
        Args:
            image: Input image tensor
            training: If False, no augmentation is applied
            
        Returns:
            Augmented image tensor
        """
        if not training or self.strategy == 'none':
            return image
        
        # Apply each augmentation based on config
        if self.config.get('horizontal_flip', False):
            image = self._random_flip_horizontal(image)
        
        if self.config.get('vertical_flip', False):
            image = self._random_flip_vertical(image)
        
        rotation_range = self.config.get('rotation_range', 0)
        if rotation_range > 0:
            image = self._random_rotation(image, rotation_range)
        
        zoom_range = self.config.get('zoom_range', 0)
        if zoom_range > 0:
            image = self._random_zoom(image, zoom_range)
        
        brightness_range = self.config.get('brightness_range', None)
        if brightness_range is not None:
            image = self._random_brightness(image, brightness_range)
        
        contrast_range = self.config.get('contrast_range', None)
        if contrast_range is not None:
            image = self._random_contrast(image, contrast_range)
        
        # Ensure values are in [0, 1] range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    @staticmethod
    def _random_flip_horizontal(image: tf.Tensor) -> tf.Tensor:
        """Randomly flip image horizontally."""
        return tf.image.random_flip_left_right(image)
    
    @staticmethod
    def _random_flip_vertical(image: tf.Tensor) -> tf.Tensor:
        """Randomly flip image vertically."""
        return tf.image.random_flip_up_down(image)
    
    @staticmethod
    def _random_rotation(image: tf.Tensor, max_rotation: float) -> tf.Tensor:
        """
        Randomly rotate image safely within tf.data pipeline.
        
        Args:
            image: Input image tensor [H, W, C]
            max_rotation: Maximum rotation in fraction of 2*pi (e.g. 0.2)
        """
        import scipy.ndimage as ndimage
        
        def _rotate_scipy(img_np):
            # max_rotation is fraction of 360 degrees
            max_angle_deg = max_rotation * 360.0
            angle = np.random.uniform(-max_angle_deg, max_angle_deg)
            
            # Scipy rotation is much more robust for Numpy arrays
            # reshape is not needed, scipy handles [H, W, C] perfectly
            rotated = ndimage.rotate(img_np, angle, reshape=False, mode='nearest')
            
            # Ensure it stays within [0, 1] range after interpolation
            rotated = np.clip(rotated, 0.0, 1.0)
            return rotated.astype(np.float32)

        # Wrap scipy rotation in numpy_function
        rotated_tensor = tf.numpy_function(func=_rotate_scipy, inp=[image], Tout=tf.float32)
        
        # Extremely crucial: restore shape information for tf.data pipeline
        rotated_tensor.set_shape(image.shape)
        
        return rotated_tensor
    
    @staticmethod
    def _random_zoom(image: tf.Tensor, zoom_range: float) -> tf.Tensor:
        """
        Randomly zoom image.
        
        Args:
            image: Input image
            zoom_range: Zoom range (0.1 means 0.9-1.1)
        """
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        
        # Random zoom factor
        zoom = tf.random.uniform([], 1.0 - zoom_range, 1.0 + zoom_range)
        
        # Calculate new dimensions
        new_height = tf.cast(tf.cast(height, tf.float32) * zoom, tf.int32)
        new_width = tf.cast(tf.cast(width, tf.float32) * zoom, tf.int32)
        
        # Resize
        image = tf.image.resize(image, [new_height, new_width])
        
        # Crop or pad to original size
        image = tf.image.resize_with_crop_or_pad(image, height, width)
        
        return image
    
    @staticmethod
    def _random_brightness(
        image: tf.Tensor,
        brightness_range: tuple
    ) -> tf.Tensor:
        """
        Randomly adjust brightness.
        
        Args:
            image: Input image
            brightness_range: (min_delta, max_delta) for brightness
        """
        delta = tf.random.uniform(
            [],
            brightness_range[0] - 1.0,
            brightness_range[1] - 1.0
        )
        return tf.image.adjust_brightness(image, delta)
    
    @staticmethod
    def _random_contrast(
        image: tf.Tensor,
        contrast_range: tuple
    ) -> tf.Tensor:
        """
        Randomly adjust contrast.
        
        Args:
            image: Input image
            contrast_range: (min_factor, max_factor) for contrast
        """
        factor = tf.random.uniform([], contrast_range[0], contrast_range[1])
        return tf.image.adjust_contrast(image, factor)
    
    def get_config(self) -> Dict[str, Any]:
        """Get augmentation configuration."""
        return {
            'strategy': self.strategy,
            'config': self.config.copy()
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DataAugmentation':
        """
        Create DataAugmentation from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            DataAugmentation instance
        """
        strategy = config.get('strategy', 'medium')
        custom_config = config.get('config', None)
        
        if custom_config and strategy == 'custom':
            return cls(strategy='medium', custom_config=custom_config)
        else:
            return cls(strategy=strategy)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DataAugmentation(strategy='{self.strategy}')"


class AdvancedAugmentation(DataAugmentation):
    """
    Advanced augmentation with additional techniques.
    
    Includes:
    - Cutout
    - Mixup (in future)
    - RandAugment (in future)
    """
    
    def __init__(
        self,
        strategy: str = 'medium',
        custom_config: Optional[Dict[str, Any]] = None,
        cutout_size: int = 16,
        cutout_prob: float = 0.5
    ):
        """
        Initialize advanced augmentation.
        
        Args:
            strategy: Base augmentation strategy
            custom_config: Custom config
            cutout_size: Size of cutout square
            cutout_prob: Probability of applying cutout
        """
        super().__init__(strategy, custom_config)
        self.cutout_size = cutout_size
        self.cutout_prob = cutout_prob
        
        logger.info(
            f"AdvancedAugmentation initialized with cutout_size={cutout_size}, "
            f"cutout_prob={cutout_prob}"
        )
    
    def apply(self, image: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Apply advanced augmentation.
        
        Args:
            image: Input image
            training: If False, no augmentation
            
        Returns:
            Augmented image
        """
        # Apply base augmentations
        image = super().apply(image, training)
        
        if not training:
            return image
        
        # Apply cutout with probability
        if tf.random.uniform([]) < self.cutout_prob:
            image = self._cutout(image, self.cutout_size)
        
        return image
    
    @staticmethod
    def _cutout(image: tf.Tensor, size: int) -> tf.Tensor:
        """
        Apply cutout augmentation.
        
        Args:
            image: Input image
            size: Size of cutout square
            
        Returns:
            Image with cutout applied
        """
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        
        # Random position
        y = tf.random.uniform([], 0, height - size, dtype=tf.int32)
        x = tf.random.uniform([], 0, width - size, dtype=tf.int32)
        
        # Create mask
        mask = tf.ones_like(image)
        cutout = tf.zeros([size, size, tf.shape(image)[2]])
        
        # Apply cutout
        mask = tf.tensor_scatter_nd_update(
            mask,
            [[i, j, k] for i in range(y, y + size) 
             for j in range(x, x + size) 
             for k in range(tf.shape(image)[2])],
            tf.reshape(cutout, [-1])
        )
        
        return image * mask
