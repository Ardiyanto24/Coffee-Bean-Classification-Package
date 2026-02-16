"""Tests for data pipeline classes."""

import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
import tempfile
import shutil

from coffee_bean_classification.data import (
    DataAugmentation,
    AdvancedAugmentation,
    CoffeeBeanDataPipeline,
)
from coffee_bean_classification.configs import DataConfig


@pytest.fixture
def sample_image():
    """Create a sample image tensor."""
    return tf.random.uniform([224, 224, 3], minval=0, maxval=1, dtype=tf.float32)


@pytest.fixture
def sample_label():
    """Create a sample one-hot label."""
    return tf.constant([0, 1, 0, 0], dtype=tf.float32)


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset directory structure."""
    temp_dir = tempfile.mkdtemp()

    # Create class directories
    classes = ["class1", "class2", "class3"]
    for class_name in classes:
        class_dir = Path(temp_dir) / class_name
        class_dir.mkdir()

        # Create dummy images (just save random arrays)
        for i in range(10):
            img_path = class_dir / f"image_{i}.jpg"
            # Create a dummy image array
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            tf.keras.preprocessing.image.save_img(str(img_path), img)

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestDataAugmentation:
    """Test DataAugmentation class."""

    def test_init_with_strategy(self):
        """Test initialization with predefined strategy."""
        aug = DataAugmentation(strategy="light")
        assert aug.strategy == "light"
        assert "horizontal_flip" in aug.config

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        custom = {"horizontal_flip": True, "rotation_range": 0.5}
        aug = DataAugmentation(custom_config=custom)
        assert aug.strategy == "custom"
        assert aug.config["rotation_range"] == 0.5

    def test_available_strategies(self):
        """Test all predefined strategies."""
        for strategy in ["none", "light", "medium", "heavy"]:
            aug = DataAugmentation(strategy=strategy)
            assert aug.strategy == strategy

    def test_apply_no_augmentation(self, sample_image):
        """Test apply with strategy='none'."""
        aug = DataAugmentation(strategy="none")
        result = aug.apply(sample_image, training=True)

        # Should return image unchanged
        assert result.shape == sample_image.shape

    def test_apply_with_training_false(self, sample_image):
        """Test that augmentation is skipped when training=False."""
        aug = DataAugmentation(strategy="heavy")
        result = aug.apply(sample_image, training=False)

        # Should return image unchanged
        assert result.shape == sample_image.shape

    def test_apply_with_augmentation(self, sample_image):
        """Test that augmentation is applied."""
        aug = DataAugmentation(strategy="medium")
        result = aug.apply(sample_image, training=True)

        # Shape should be preserved
        assert result.shape == sample_image.shape

        # Values should be in [0, 1]
        assert tf.reduce_min(result) >= 0.0
        assert tf.reduce_max(result) <= 1.0

    def test_horizontal_flip(self, sample_image):
        """Test horizontal flip augmentation."""
        flipped = DataAugmentation._random_flip_horizontal(sample_image)
        assert flipped.shape == sample_image.shape

    def test_vertical_flip(self, sample_image):
        """Test vertical flip augmentation."""
        flipped = DataAugmentation._random_flip_vertical(sample_image)
        assert flipped.shape == sample_image.shape

    def test_random_zoom(self, sample_image):
        """Test random zoom augmentation."""
        zoomed = DataAugmentation._random_zoom(sample_image, zoom_range=0.1)
        assert zoomed.shape == sample_image.shape

    def test_random_brightness(self, sample_image):
        """Test brightness adjustment."""
        adjusted = DataAugmentation._random_brightness(sample_image, brightness_range=(0.8, 1.2))
        assert adjusted.shape == sample_image.shape

    def test_random_contrast(self, sample_image):
        """Test contrast adjustment."""
        adjusted = DataAugmentation._random_contrast(sample_image, contrast_range=(0.8, 1.2))
        assert adjusted.shape == sample_image.shape

    def test_get_config(self):
        """Test get_config method."""
        aug = DataAugmentation(strategy="medium")
        config = aug.get_config()

        assert "strategy" in config
        assert "config" in config
        assert config["strategy"] == "medium"

    def test_from_config(self):
        """Test from_config method."""
        config = {"strategy": "heavy", "config": {"horizontal_flip": True}}
        aug = DataAugmentation.from_config(config)
        assert aug.strategy == "heavy"


class TestAdvancedAugmentation:
    """Test AdvancedAugmentation class."""

    def test_init(self):
        """Test initialization."""
        aug = AdvancedAugmentation(strategy="medium", cutout_size=20, cutout_prob=0.5)
        assert aug.strategy == "medium"
        assert aug.cutout_size == 20
        assert aug.cutout_prob == 0.5

    def test_apply_with_cutout(self, sample_image):
        """Test apply with cutout."""
        aug = AdvancedAugmentation(
            strategy="light", cutout_size=16, cutout_prob=1.0  # Always apply
        )
        result = aug.apply(sample_image, training=True)

        assert result.shape == sample_image.shape


class TestCoffeeBeanDataPipeline:
    """Test CoffeeBeanDataPipeline class."""

    def test_init(self, temp_dataset):
        """Test initialization."""
        config = DataConfig(dataset_path=temp_dataset)
        pipeline = CoffeeBeanDataPipeline(config)

        assert pipeline.config == config
        assert pipeline.augmentation is not None

    def test_load_dataset(self, temp_dataset):
        """Test dataset loading."""
        config = DataConfig(dataset_path=temp_dataset, batch_size=4, split_ratio=(0.6, 0.2, 0.2))
        pipeline = CoffeeBeanDataPipeline(config)

        train_ds, val_ds, test_ds = pipeline.load_dataset()

        # Check datasets are created
        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None

        # Check class names detected
        assert len(pipeline.get_class_names()) == 3
        assert pipeline.get_num_classes() == 3

    def test_preprocess(self, temp_dataset):
        """Test preprocessing."""
        config = DataConfig(dataset_path=temp_dataset)
        pipeline = CoffeeBeanDataPipeline(config)

        # Create sample image (0-255 range)
        image = tf.random.uniform([224, 224, 3], minval=0, maxval=255)
        label = tf.constant([1, 0, 0], dtype=tf.float32)

        processed_img, processed_label = pipeline.preprocess(image, label)

        # Check normalization
        assert tf.reduce_min(processed_img) >= 0.0
        assert tf.reduce_max(processed_img) <= 1.0

        # Label unchanged
        assert tf.reduce_all(processed_label == label)

    def test_get_dataset_info(self, temp_dataset):
        """Test get_dataset_info method."""
        config = DataConfig(dataset_path=temp_dataset, batch_size=8)
        pipeline = CoffeeBeanDataPipeline(config)
        pipeline.load_dataset()

        info = pipeline.get_dataset_info()

        assert "num_classes" in info
        assert "class_names" in info
        assert "image_size" in info
        assert "batch_size" in info
        assert info["batch_size"] == 8
        assert info["num_classes"] == 3

    def test_get_class_distribution(self, temp_dataset):
        """Test class distribution calculation."""
        config = DataConfig(dataset_path=temp_dataset)
        pipeline = CoffeeBeanDataPipeline(config)
        pipeline.load_dataset()

        distribution = pipeline.get_class_distribution()

        assert isinstance(distribution, dict)
        assert len(distribution) == 3
        assert all(count > 0 for count in distribution.values())

    def test_get_sample_batch(self, temp_dataset):
        """Test getting a sample batch."""
        config = DataConfig(dataset_path=temp_dataset, batch_size=4)
        pipeline = CoffeeBeanDataPipeline(config)
        pipeline.load_dataset()

        images, labels = pipeline.get_sample_batch("train")

        # Check batch shape
        assert images.shape[0] == 4  # batch_size
        assert images.shape[1:] == (224, 224, 3)  # image_size + channels

        # Check labels shape
        assert labels.shape == (4, 3)  # batch_size x num_classes

    def test_augmentation_integration(self, temp_dataset):
        """Test that augmentation is properly integrated."""
        config = DataConfig(
            dataset_path=temp_dataset,
            augmentation_params={"horizontal_flip": True, "rotation_range": 0.1},
        )
        pipeline = CoffeeBeanDataPipeline(config)
        pipeline.load_dataset()

        # Augmentation should be applied to training set
        assert pipeline.augmentation is not None
        assert pipeline.augmentation.config["horizontal_flip"] == True

    def test_repr(self, temp_dataset):
        """Test string representation."""
        config = DataConfig(dataset_path=temp_dataset)
        pipeline = CoffeeBeanDataPipeline(config)
        pipeline.load_dataset()

        repr_str = repr(pipeline)

        assert "CoffeeBeanDataPipeline" in repr_str
        assert "classes=3" in repr_str
