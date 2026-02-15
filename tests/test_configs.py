"""Tests for configuration classes."""

import pytest
import tempfile
from pathlib import Path

from coffee_bean_classification.configs import (
    DataConfig,
    ModelConfig,
    TrainingConfig
)
from coffee_bean_classification.utils import ValidationError


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_init_default(self, tmp_path):
        """Test initialization with default parameters."""
        config = DataConfig(dataset_path=str(tmp_path))
        assert config.batch_size == 32
        assert config.image_size == (224, 224)
        assert config.split_ratio == (0.7, 0.15, 0.15)
    
    def test_init_custom(self, tmp_path):
        """Test initialization with custom parameters."""
        config = DataConfig(
            dataset_path=str(tmp_path),
            image_size=(300, 300),
            batch_size=64,
            split_ratio=(0.8, 0.1, 0.1)
        )
        assert config.batch_size == 64
        assert config.image_size == (300, 300)
        assert config.split_ratio == (0.8, 0.1, 0.1)
    
    def test_image_size_from_int(self, tmp_path):
        """Test image size from single integer."""
        config = DataConfig(dataset_path=str(tmp_path), image_size=256)
        assert config.image_size == (256, 256)
    
    def test_validation_success(self, tmp_path):
        """Test successful validation."""
        config = DataConfig(dataset_path=str(tmp_path))
        assert config.validate() == True
    
    def test_validation_invalid_ratio(self, tmp_path):
        """Test validation with invalid split ratio."""
        config = DataConfig(
            dataset_path=str(tmp_path),
            split_ratio=(0.6, 0.3, 0.2)  # Sum > 1.0
        )
        with pytest.raises(ValidationError):
            config.validate()
    
    def test_to_dict(self, tmp_path):
        """Test conversion to dictionary."""
        config = DataConfig(dataset_path=str(tmp_path))
        config_dict = config.to_dict()
        assert 'dataset_path' in config_dict
        assert 'batch_size' in config_dict
        assert config_dict['batch_size'] == 32
    
    def test_save_load_yaml(self, tmp_path):
        """Test save and load YAML."""
        config = DataConfig(dataset_path=str(tmp_path), batch_size=64)
        yaml_path = tmp_path / 'config.yaml'
        
        config.save(yaml_path)
        loaded_config = DataConfig.load(yaml_path)
        
        assert loaded_config.batch_size == 64
        assert loaded_config.dataset_path == str(tmp_path)
    
    def test_save_load_json(self, tmp_path):
        """Test save and load JSON."""
        config = DataConfig(dataset_path=str(tmp_path), batch_size=64)
        json_path = tmp_path / 'config.json'
        
        config.save(json_path)
        loaded_config = DataConfig.load(json_path)
        
        assert loaded_config.batch_size == 64


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        config = ModelConfig(architecture='resnet50')
        assert config.input_shape == (224, 224, 3)
        assert config.num_classes == 4
        assert config.dropout_rate == 0.2
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        config = ModelConfig(
            architecture='efficientnet_b3',
            input_shape=(300, 300, 3),
            num_classes=10,
            dropout_rate=0.3
        )
        assert config.input_shape == (300, 300, 3)
        assert config.num_classes == 10
        assert config.dropout_rate == 0.3
    
    def test_validation_success(self):
        """Test successful validation."""
        config = ModelConfig(architecture='resnet50')
        assert config.validate() == True
    
    def test_validation_invalid_architecture(self):
        """Test validation with invalid architecture."""
        config = ModelConfig(architecture='invalid_arch')
        with pytest.raises(ValidationError):
            config.validate()
    
    def test_validation_invalid_dropout(self):
        """Test validation with invalid dropout rate."""
        config = ModelConfig(architecture='resnet50', dropout_rate=1.5)
        with pytest.raises(ValidationError):
            config.validate()
    
    def test_for_architecture(self):
        """Test for_architecture factory method."""
        config = ModelConfig.for_architecture('efficientnet_b3', num_classes=10)
        assert config.architecture == 'efficientnet_b3'
        assert config.input_shape == (300, 300, 3)
        assert config.num_classes == 10
    
    def test_get_input_size(self):
        """Test get_input_size method."""
        config = ModelConfig(architecture='resnet50', input_shape=(256, 256, 3))
        assert config.get_input_size() == (256, 256)
    
    def test_is_transfer_learning(self):
        """Test is_transfer_learning method."""
        config1 = ModelConfig(architecture='resnet50', weights='imagenet')
        assert config1.is_transfer_learning() == True
        
        config2 = ModelConfig(architecture='resnet50', weights=None)
        assert config2.is_transfer_learning() == False


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_init_default(self):
        """Test initialization with default parameters."""
        config = TrainingConfig()
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.seed == 42
    
    def test_init_custom(self):
        """Test initialization with custom parameters."""
        config = TrainingConfig(
            epochs=100,
            batch_size=64,
            learning_rate=0.0001,
            seed=123
        )
        assert config.epochs == 100
        assert config.batch_size == 64
        assert config.learning_rate == 0.0001
        assert config.seed == 123
    
    def test_validation_success(self):
        """Test successful validation."""
        config = TrainingConfig()
        assert config.validate() == True
    
    def test_validation_invalid_epochs(self):
        """Test validation with invalid epochs."""
        config = TrainingConfig(epochs=0)
        with pytest.raises(ValidationError):
            config.validate()
    
    def test_validation_invalid_lr(self):
        """Test validation with invalid learning rate."""
        config = TrainingConfig(learning_rate=1.5)
        with pytest.raises(ValidationError):
            config.validate()
    
    def test_nested_configs(self, tmp_path):
        """Test with nested data and model configs."""
        data_config = DataConfig(dataset_path=str(tmp_path))
        model_config = ModelConfig(architecture='resnet50')
        
        training_config = TrainingConfig(
            data_config=data_config,
            model_config=model_config
        )
        
        assert training_config.data_config is not None
        assert training_config.model_config is not None
        assert training_config.validate() == True
    
    def test_quick_start(self, tmp_path):
        """Test quick_start factory method."""
        config = TrainingConfig.quick_start(
            dataset_path=str(tmp_path),
            architecture='efficientnet_b0',
            num_classes=4,
            epochs=30
        )
        
        assert config.epochs == 30
        assert config.data_config is not None
        assert config.model_config is not None
        assert config.model_config.architecture == 'efficientnet_b0'
    
    def test_save_load_nested(self, tmp_path):
        """Test save and load with nested configs."""
        data_config = DataConfig(dataset_path=str(tmp_path))
        model_config = ModelConfig(architecture='resnet50')
        
        config = TrainingConfig(
            epochs=100,
            data_config=data_config,
            model_config=model_config
        )
        
        yaml_path = tmp_path / 'training_config.yaml'
        config.save(yaml_path)
        
        loaded_config = TrainingConfig.load(yaml_path)
        
        assert loaded_config.epochs == 100
        assert loaded_config.data_config is not None
        assert loaded_config.model_config is not None
    
    def test_callback_management(self):
        """Test callback enable/disable methods."""
        config = TrainingConfig()
        
        config.disable_callback('early_stopping')
        assert config.callbacks['early_stopping'] == False
        
        config.enable_callback('early_stopping')
        assert config.callbacks['early_stopping'] == True
