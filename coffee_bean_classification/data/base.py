"""Base data pipeline class for all data processing pipelines."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import tensorflow as tf

from ..utils import get_logger

logger = get_logger(__name__)


class BaseDataPipeline(ABC):
    """
    Abstract base class for data pipelines.
    
    All data pipeline classes must inherit from this base class and implement
    the required abstract methods.
    
    Attributes:
        config: Data configuration object
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset
        
    Example:
        >>> class MyDataPipeline(BaseDataPipeline):
        ...     def load_dataset(self):
        ...         # Implementation
        ...         pass
        ...     
        ...     def preprocess(self, image):
        ...         # Implementation
        ...         pass
    """
    
    def __init__(self, config):
        """
        Initialize base data pipeline.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self._train_ds: Optional[tf.data.Dataset] = None
        self._val_ds: Optional[tf.data.Dataset] = None
        self._test_ds: Optional[tf.data.Dataset] = None
        self._class_names: Optional[list] = None
        self._num_classes: Optional[int] = None
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load dataset and create train/val/test splits.
        
        This method must be implemented by all subclasses.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess a single image and label.
        
        This method must be implemented by all subclasses.
        
        Args:
            image: Input image tensor
            label: Input label tensor
            
        Returns:
            Tuple of (preprocessed_image, label)
        """
        pass
    
    def augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply data augmentation to image.
        
        Default implementation returns image unchanged.
        Override this method to implement custom augmentation.
        
        Args:
            image: Input image tensor
            label: Input label tensor
            
        Returns:
            Tuple of (augmented_image, label)
        """
        return image, label
    
    def get_datasets(
        self,
        load_if_needed: bool = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get train, validation, and test datasets.
        
        Args:
            load_if_needed: If True, load datasets if not already loaded
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if load_if_needed and self._train_ds is None:
            logger.info("Datasets not loaded, loading now...")
            self.load_dataset()
        
        return self._train_ds, self._val_ds, self._test_ds
    
    def get_class_names(self) -> list:
        """
        Get list of class names.
        
        Returns:
            List of class names
        """
        if self._class_names is None:
            logger.warning("Class names not set")
            return []
        return self._class_names
    
    def get_num_classes(self) -> int:
        """
        Get number of classes.
        
        Returns:
            Number of classes
        """
        if self._num_classes is None:
            if self._class_names is not None:
                self._num_classes = len(self._class_names)
            else:
                logger.warning("Number of classes not determined")
                return 0
        return self._num_classes
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'num_classes': self.get_num_classes(),
            'class_names': self.get_class_names(),
            'image_size': self.config.image_size,
            'batch_size': self.config.batch_size,
        }
        
        # Add dataset sizes if available
        if self._train_ds is not None:
            try:
                info['train_size'] = tf.data.experimental.cardinality(self._train_ds).numpy()
            except:
                info['train_size'] = 'unknown'
        
        if self._val_ds is not None:
            try:
                info['val_size'] = tf.data.experimental.cardinality(self._val_ds).numpy()
            except:
                info['val_size'] = 'unknown'
        
        if self._test_ds is not None:
            try:
                info['test_size'] = tf.data.experimental.cardinality(self._test_ds).numpy()
            except:
                info['test_size'] = 'unknown'
        
        return info
    
    def __repr__(self) -> str:
        """String representation of data pipeline."""
        info = self.get_dataset_info()
        return (
            f"{self.__class__.__name__}("
            f"classes={info['num_classes']}, "
            f"image_size={info['image_size']}, "
            f"batch_size={info['batch_size']})"
        )
