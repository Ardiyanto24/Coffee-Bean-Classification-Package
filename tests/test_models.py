"""Tests for model factory and model implementations."""

import pytest
import tensorflow as tf

from coffee_bean_classification.models import (
    ModelFactory,
    ResNet50Model,
    EfficientNetB0Model,
    EfficientNetB3Model,
    MobileNetV3Model,
    DenseNet121Model,
    BaseModel
)
from coffee_bean_classification.configs import ModelConfig
from coffee_bean_classification.utils import ValidationError


@pytest.fixture
def basic_config():
    """Create a basic model configuration."""
    return ModelConfig(
        architecture='resnet50',
        input_shape=(224, 224, 3),
        num_classes=4,
        weights=None,  # Don't load pretrained weights for testing
        dropout_rate=0.2
    )


class TestModelFactory:
    """Test ModelFactory class."""
    
    def test_list_available(self):
        """Test listing available models."""
        available = ModelFactory.list_available()
        
        assert isinstance(available, list)
        assert len(available) == 5
        assert 'resnet50' in available
        assert 'efficientnet_b0' in available
        assert 'efficientnet_b3' in available
        assert 'mobilenet_v3' in available
        assert 'densenet121' in available
    
    def test_create_resnet50(self, basic_config):
        """Test creating ResNet50 model."""
        model = ModelFactory.create('resnet50', basic_config)
        
        assert isinstance(model, ResNet50Model)
        assert isinstance(model, BaseModel)
        assert model.config == basic_config
    
    def test_create_efficientnet_b0(self, basic_config):
        """Test creating EfficientNetB0 model."""
        model = ModelFactory.create('efficientnet_b0', basic_config)
        
        assert isinstance(model, EfficientNetB0Model)
        assert isinstance(model, BaseModel)
    
    def test_create_efficientnet_b3(self, basic_config):
        """Test creating EfficientNetB3 model."""
        model = ModelFactory.create('efficientnet_b3', basic_config)
        
        assert isinstance(model, EfficientNetB3Model)
        assert isinstance(model, BaseModel)
    
    def test_create_mobilenet_v3(self, basic_config):
        """Test creating MobileNetV3 model."""
        model = ModelFactory.create('mobilenet_v3', basic_config)
        
        assert isinstance(model, MobileNetV3Model)
        assert isinstance(model, BaseModel)
    
    def test_create_densenet121(self, basic_config):
        """Test creating DenseNet121 model."""
        model = ModelFactory.create('densenet121', basic_config)
        
        assert isinstance(model, DenseNet121Model)
        assert isinstance(model, BaseModel)
    
    def test_create_invalid_model(self, basic_config):
        """Test creating invalid model raises error."""
        with pytest.raises(ValidationError):
            ModelFactory.create('invalid_model', basic_config)
    
    def test_case_insensitive(self, basic_config):
        """Test that model names are case insensitive."""
        model1 = ModelFactory.create('ResNet50', basic_config)
        model2 = ModelFactory.create('RESNET50', basic_config)
        model3 = ModelFactory.create('resnet50', basic_config)
        
        assert type(model1) == type(model2) == type(model3)
    
    def test_register_custom_model(self, basic_config):
        """Test registering a custom model."""
        @ModelFactory.register('custom_test_model')
        class CustomTestModel(BaseModel):
            def build(self):
                inputs = tf.keras.Input(shape=self.config.input_shape)
                outputs = tf.keras.layers.Dense(self.config.num_classes)(inputs)
                return tf.keras.Model(inputs, outputs)
        
        # Check it's registered
        assert 'custom_test_model' in ModelFactory.list_available()
        
        # Check we can create it
        model = ModelFactory.create('custom_test_model', basic_config)
        assert isinstance(model, CustomTestModel)
        
        # Cleanup
        ModelFactory.unregister('custom_test_model')
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = ModelFactory.get_model_info('resnet50')
        
        assert 'name' in info
        assert 'class_name' in info
        assert 'module' in info
        assert info['name'] == 'resnet50'
        assert info['class_name'] == 'ResNet50Model'
    
    def test_is_registered(self):
        """Test checking if model is registered."""
        assert ModelFactory.is_registered('resnet50') == True
        assert ModelFactory.is_registered('invalid_model') == False
    
    def test_compare_models(self):
        """Test model comparison."""
        comparison = ModelFactory.compare_models()
        
        assert isinstance(comparison, dict)
        assert len(comparison) == 5
        assert 'resnet50' in comparison
        assert 'approximate_params' in comparison['resnet50']
        assert 'optimal_input_size' in comparison['resnet50']


class TestResNet50Model:
    """Test ResNet50Model class."""
    
    def test_init(self, basic_config):
        """Test initialization."""
        model = ResNet50Model(basic_config)
        
        assert model.config == basic_config
        assert model.model is None
        assert model.backbone is None
    
    def test_build(self, basic_config):
        """Test building the model."""
        model = ResNet50Model(basic_config)
        keras_model = model.build()
        
        assert keras_model is not None
        assert isinstance(keras_model, tf.keras.Model)
        assert keras_model.name == 'ResNet50'
        
        # Check input shape
        assert keras_model.input_shape == (None, 224, 224, 3)
        
        # Check output shape
        assert keras_model.output_shape == (None, 4)
    
    def test_count_parameters(self, basic_config):
        """Test parameter counting."""
        model = ResNet50Model(basic_config)
        model.build()
        
        params = model.count_parameters()
        
        assert 'total' in params
        assert 'trainable' in params
        assert 'non_trainable' in params
        assert params['total'] > 0
        assert params['trainable'] + params['non_trainable'] == params['total']
    
    def test_freeze_unfreeze(self, basic_config):
        """Test freezing and unfreezing backbone."""
        model = ResNet50Model(basic_config)
        model.build()
        
        # Unfreeze
        model.unfreeze_backbone()
        assert model.backbone.trainable == True
        
        # Freeze
        model.freeze_backbone()
        assert model.backbone.trainable == False


class TestEfficientNetModels:
    """Test EfficientNet model classes."""
    
    def test_efficientnet_b0_build(self, basic_config):
        """Test EfficientNetB0 building."""
        model = EfficientNetB0Model(basic_config)
        keras_model = model.build()
        
        assert keras_model is not None
        assert keras_model.name == 'EfficientNetB0'
        assert keras_model.output_shape == (None, 4)
    
    def test_efficientnet_b3_build(self, basic_config):
        """Test EfficientNetB3 building."""
        # B3 uses 300x300 input
        config = ModelConfig(
            architecture='efficientnet_b3',
            input_shape=(300, 300, 3),
            num_classes=4,
            weights=None
        )
        model = EfficientNetB3Model(config)
        keras_model = model.build()
        
        assert keras_model is not None
        assert keras_model.name == 'EfficientNetB3'
        assert keras_model.input_shape == (None, 300, 300, 3)


class TestMobileNetV3Model:
    """Test MobileNetV3Model class."""
    
    def test_build(self, basic_config):
        """Test building the model."""
        model = MobileNetV3Model(basic_config)
        keras_model = model.build()
        
        assert keras_model is not None
        assert keras_model.name == 'MobileNetV3Small'
        assert keras_model.output_shape == (None, 4)
    
    def test_model_size(self, basic_config):
        """Test getting model size."""
        model = MobileNetV3Model(basic_config)
        model.build()
        
        size_mb = model.get_model_size_mb()
        
        assert size_mb > 0
        assert size_mb < 100  # MobileNet should be small


class TestDenseNet121Model:
    """Test DenseNet121Model class."""
    
    def test_build(self, basic_config):
        """Test building the model."""
        model = DenseNet121Model(basic_config)
        keras_model = model.build()
        
        assert keras_model is not None
        assert keras_model.name == 'DenseNet121'
        assert keras_model.output_shape == (None, 4)
    
    def test_additional_dense_layer(self, basic_config):
        """Test that DenseNet has additional dense layer."""
        model = DenseNet121Model(basic_config)
        keras_model = model.build()
        
        # Check for intermediate dense layer
        layer_names = [layer.name for layer in keras_model.layers]
        assert 'dense_intermediate' in layer_names
    
    def test_freeze_bn_layers(self, basic_config):
        """Test freezing batch normalization layers."""
        model = DenseNet121Model(basic_config)
        model.build()
        
        # Unfreeze all first
        model.unfreeze_backbone()
        
        # Then freeze BN layers
        model.freeze_bn_layers()
        
        # Check that BN layers are frozen
        for layer in model.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert layer.trainable == False


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_all_models_can_be_created(self, basic_config):
        """Test that all registered models can be created."""
        for model_name in ModelFactory.list_available():
            model = ModelFactory.create(model_name, basic_config)
            assert model is not None
            assert isinstance(model, BaseModel)
    
    def test_model_forward_pass(self, basic_config):
        """Test forward pass through model."""
        model = ResNet50Model(basic_config)
        keras_model = model.build()
        
        # Create dummy input
        dummy_input = tf.random.normal([1, 224, 224, 3])
        
        # Forward pass
        output = keras_model(dummy_input, training=False)
        
        # Check output shape
        assert output.shape == (1, 4)
        
        # Check output is probability distribution
        assert tf.reduce_all(output >= 0)
        assert tf.reduce_all(output <= 1)
        # Sum should be approximately 1 (softmax)
        assert tf.abs(tf.reduce_sum(output) - 1.0) < 0.01
    
    def test_model_compile(self, basic_config):
        """Test compiling models."""
        model = ModelFactory.create('resnet50', basic_config)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        assert model._compiled == True
        assert model.model is not None
