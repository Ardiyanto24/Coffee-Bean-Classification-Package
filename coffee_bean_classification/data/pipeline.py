"""Coffee Bean dataset pipeline implementation."""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt

from .base import BaseDataPipeline
from .augmentation import DataAugmentation
from ..utils import get_logger, validate_path

logger = get_logger(__name__)


class CoffeeBeanDataPipeline(BaseDataPipeline):
    """
    Data pipeline for Coffee Bean classification dataset.
    
    Handles:
    - Loading images from directory structure
    - Train/val/test splitting
    - Preprocessing and normalization
    - Data augmentation
    - TensorFlow dataset creation
    
    Example:
        >>> from coffee_bean_classification.configs import DataConfig
        >>> config = DataConfig(dataset_path='/path/to/dataset')
        >>> pipeline = CoffeeBeanDataPipeline(config)
        >>> train_ds, val_ds, test_ds = pipeline.load_dataset()
    """
    
    def __init__(self, config):
        """
        Initialize coffee bean data pipeline.
        
        Args:
            config: DataConfig instance
        """
        super().__init__(config)
        
        # Validate dataset path
        validate_path(self.config.dataset_path, must_exist=True, path_type='dir')
        
        # Initialize augmentation
        self.augmentation = DataAugmentation(
            strategy='medium',
            custom_config=self.config.augmentation_params
        )
        
        # AUTOTUNE for performance
        self.AUTOTUNE = tf.data.AUTOTUNE
        
        logger.info(f"CoffeeBeanDataPipeline initialized for {self.config.dataset_path}")
    
    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load dataset from directory and create splits.
        
        Expected directory structure:
        dataset_path/
            class1/
                image1.jpg
                image2.jpg
            class2/
                image1.jpg
                image2.jpg
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Loading dataset from directory...")
        
        # Load full dataset
        dataset_path = Path(self.config.dataset_path)
        
        # Get class names from directory structure
        class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        self._class_names = [d.name for d in class_dirs]
        self._num_classes = len(self._class_names)
        
        logger.info(f"Found {self._num_classes} classes: {self._class_names}")
        
        # Load using image_dataset_from_directory
        full_dataset = tf.keras.utils.image_dataset_from_directory(
            str(dataset_path),
            labels='inferred',
            label_mode='categorical',
            class_names=self._class_names,
            batch_size=None,  # Unbatched for splitting
            image_size=self.config.image_size,
            shuffle=True,
            seed=42
        )
        
        # Get dataset size
        dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
        logger.info(f"Total images: {dataset_size}")
        
        # Calculate split sizes
        train_size = int(self.config.train_ratio * dataset_size)
        val_size = int(self.config.val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        logger.info(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Create splits
        train_dataset = full_dataset.take(train_size)
        remaining = full_dataset.skip(train_size)
        val_dataset = remaining.take(val_size)
        test_dataset = remaining.skip(val_size)
        
        # Configure datasets
        self._train_ds = self._configure_dataset(
            train_dataset,
            is_training=True,
            cache=True
        )
        
        self._val_ds = self._configure_dataset(
            val_dataset,
            is_training=False,
            cache=True
        )
        
        self._test_ds = self._configure_dataset(
            test_dataset,
            is_training=False,
            cache=False
        )
        
        logger.info("✓ Datasets loaded and configured successfully")
        
        return self._train_ds, self._val_ds, self._test_ds
    
    def preprocess(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess image and label.
        
        Args:
            image: Input image tensor (0-255)
            label: One-hot encoded label
            
        Returns:
            Tuple of (preprocessed_image, label)
        """
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        return image, label
    
    def augment(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply data augmentation.
        
        Args:
            image: Preprocessed image tensor
            label: One-hot encoded label
            
        Returns:
            Tuple of (augmented_image, label)
        """
        image = self.augmentation.apply(image, training=True)
        return image, label
    
    def _configure_dataset(
        self,
        dataset: tf.data.Dataset,
        is_training: bool = True,
        cache: bool = True
    ) -> tf.data.Dataset:
        """
        Configure dataset with preprocessing, augmentation, batching, etc.
        
        Args:
            dataset: Input dataset
            is_training: Whether this is training dataset
            cache: Whether to cache dataset
            
        Returns:
            Configured dataset
        """
        # Preprocess
        dataset = dataset.map(
            self.preprocess,
            num_parallel_calls=self.AUTOTUNE
        )
        
        # Cache before augmentation for efficiency
        if cache:
            if self.config.cache_dir:
                cache_path = Path(self.config.cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                dataset = dataset.cache(str(cache_path / 'cache'))
            else:
                dataset = dataset.cache()
        
        # Shuffle for training
        if is_training:
            dataset = dataset.shuffle(
                buffer_size=self.config.shuffle_buffer_size,
                reshuffle_each_iteration=True
            )
            
            # Apply augmentation
            dataset = dataset.map(
                self.augment,
                num_parallel_calls=self.AUTOTUNE
            )
        
        # Batch
        dataset = dataset.batch(self.config.batch_size)
        
        # Prefetch
        if self.config.prefetch_buffer_size == -1:
            dataset = dataset.prefetch(self.AUTOTUNE)
        else:
            dataset = dataset.prefetch(self.config.prefetch_buffer_size)
        
        return dataset
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of classes in the dataset.
        
        Returns:
            Dictionary mapping class names to counts
        """
        if self._train_ds is None:
            logger.warning("Dataset not loaded yet")
            return {}
        
        logger.info("Calculating class distribution...")
        
        # Count classes in training set
        class_counts = {name: 0 for name in self._class_names}
        
        # Unbatch and count
        unbatched = self._train_ds.unbatch()
        for _, label in unbatched:
            class_idx = tf.argmax(label).numpy()
            class_name = self._class_names[class_idx]
            class_counts[class_name] += 1
        
        logger.info(f"Class distribution: {class_counts}")
        return class_counts
    
    def visualize_samples(
        self,
        n: int = 9,
        dataset: str = 'train',
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize random samples from dataset.
        
        Args:
            n: Number of samples to show (should be perfect square)
            dataset: Which dataset to visualize ('train', 'val', 'test')
            save_path: Optional path to save figure
        """
        # Select dataset
        if dataset == 'train':
            ds = self._train_ds
        elif dataset == 'val':
            ds = self._val_ds
        elif dataset == 'test':
            ds = self._test_ds
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        if ds is None:
            logger.error(f"{dataset} dataset not loaded")
            return
        
        # Get samples
        samples = ds.unbatch().take(n)
        
        # Calculate grid size
        grid_size = int(np.sqrt(n))
        if grid_size * grid_size < n:
            grid_size += 1
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'Sample Images from {dataset.capitalize()} Set', fontsize=16)
        
        # Plot samples
        for idx, (image, label) in enumerate(samples):
            if idx >= n:
                break
            
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col] if grid_size > 1 else axes[idx]
            
            # Display image
            ax.imshow(image.numpy())
            
            # Get class name
            class_idx = tf.argmax(label).numpy()
            class_name = self._class_names[class_idx]
            
            ax.set_title(class_name)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n, grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col] if grid_size > 1 else axes[idx]
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def show_augmentation_examples(
        self,
        image_idx: int = 0,
        n_examples: int = 6,
        save_path: Optional[str] = None
    ) -> None:
        """
        Show augmentation examples for a single image.
        
        Args:
            image_idx: Index of image to augment
            n_examples: Number of augmented versions to show
            save_path: Optional path to save figure
        """
        if self._train_ds is None:
            logger.error("Training dataset not loaded")
            return
        
        # Get original image
        for idx, (image, label) in enumerate(self._train_ds.unbatch()):
            if idx == image_idx:
                original_image = image
                original_label = label
                break
        
        # Create figure
        fig, axes = plt.subplots(2, (n_examples + 1) // 2, figsize=(15, 6))
        fig.suptitle('Data Augmentation Examples', fontsize=16)
        axes = axes.flatten()
        
        # Get class name
        class_idx = tf.argmax(original_label).numpy()
        class_name = self._class_names[class_idx]
        
        # Show original
        axes[0].imshow(original_image.numpy())
        axes[0].set_title(f'Original - {class_name}')
        axes[0].axis('off')
        
        # Show augmented versions
        for i in range(1, n_examples):
            augmented, _ = self.augment(original_image, original_label)
            axes[i].imshow(augmented.numpy())
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved augmentation examples to {save_path}")
        
        plt.show()
    
    def get_sample_batch(
        self,
        dataset: str = 'train'
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get a single batch from dataset.
        
        Args:
            dataset: Which dataset ('train', 'val', 'test')
            
        Returns:
            Tuple of (images, labels) batch
        """
        if dataset == 'train':
            ds = self._train_ds
        elif dataset == 'val':
            ds = self._val_ds
        elif dataset == 'test':
            ds = self._test_ds
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        if ds is None:
            raise ValueError(f"{dataset} dataset not loaded")
        
        # Get first batch
        for images, labels in ds.take(1):
            return images, labels