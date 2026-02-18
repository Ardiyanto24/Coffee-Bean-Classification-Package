"""Integration tests for the complete pipeline."""

import pytest
import tensorflow as tf
import numpy as np
import tempfile
import shutil
from pathlib import Path

from coffee_bean_classification.pipeline import CoffeeBeanPipeline
from coffee_bean_classification.configs import TrainingConfig, DataConfig, ModelConfig
from coffee_bean_classification.data import CoffeeBeanDataPipeline
from coffee_bean_classification.models import ModelFactory
from coffee_bean_classification.training import ModelTrainer
from coffee_bean_classification.evaluation import ClassificationEvaluator
from coffee_bean_classification.registry import ModelRegistry, ModelMetadata


@pytest.fixture
def temp_dataset():
    """Create temporary dataset for testing."""
    temp_dir = tempfile.mkdtemp()

    # Create class directories
    classes = ["class_a", "class_b", "class_c", "class_d"]
    for class_name in classes:
        class_dir = Path(temp_dir) / class_name
        class_dir.mkdir()

        # Create 20 dummy images per class
        for i in range(20):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = class_dir / f"image_{i}.jpg"
            tf.keras.preprocessing.image.save_img(str(img_path), img)

    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_output():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDataPipelineIntegration:
    """Test data pipeline integration."""

    def test_data_pipeline_with_config(self, temp_dataset):
        """Test data pipeline with configuration."""
        config = DataConfig(dataset_path=temp_dataset, batch_size=4, split_ratio=(0.6, 0.2, 0.2))

        pipeline = CoffeeBeanDataPipeline(config)
        train_ds, val_ds, test_ds = pipeline.load_dataset()

        # Verify datasets created
        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None

        # Verify class detection
        assert pipeline.get_num_classes() == 4
        assert len(pipeline.get_class_names()) == 4


class TestModelTrainingIntegration:
    """Test model training integration."""

    def test_train_single_model_flow(self, temp_dataset, temp_output):
        """Test complete single model training flow."""
        # Create configs
        data_config = DataConfig(dataset_path=temp_dataset, batch_size=4)

        model_config = ModelConfig(
            architecture="resnet50", # <--- FIX: Diubah dari mobilenet_v3
            input_shape=(224, 224, 3),
            num_classes=4,
            weights=None,  # Don't load pretrained for speed
        )

        training_config = TrainingConfig(
            epochs=2,  # Short for testing
            batch_size=4,
            data_config=data_config,
            model_config=model_config,
            output_dir=temp_output,
            verbose=0,
        )

        # Train
        trainer = ModelTrainer(training_config)
        history = trainer.train_single_model("resnet50") # <--- FIX: Diubah dari mobilenet_v3

        # Verify training completed
        assert history is not None
        assert "loss" in history.history
        assert "accuracy" in history.history


class TestEvaluationIntegration:
    """Test evaluation integration."""

    def test_evaluation_flow(self, temp_dataset):
        """Test complete evaluation flow."""
        # Create simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(4, activation="softmax"),
            ]
        )

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Create dummy dataset
        x = tf.random.normal([20, 224, 224, 3])
        y = tf.keras.utils.to_categorical(np.random.randint(0, 4, 20), num_classes=4)
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)

        # Evaluate
        evaluator = ClassificationEvaluator(model, class_names=["A", "B", "C", "D"])

        metrics = evaluator.evaluate(dataset, verbose=0)

        # Verify metrics calculated
        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics


class TestRegistryIntegration:
    """Test model registry integration."""

    def test_register_and_load_model(self, temp_output):
        """Test model registration and loading."""
        # Create simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )

        # Create metadata
        from datetime import datetime

        metadata = ModelMetadata(
            model_name="test_model",
            version="1.0",
            architecture="simple",
            created_at=datetime.now().isoformat(),
            metrics={"accuracy": 0.95},
            config={},
            input_shape=(5,),
            num_classes=3,
            tags=["test"],
        )

        # Register
        registry = ModelRegistry(temp_output)
        model_id = registry.register_model(model, metadata)

        # Load
        loaded_model = registry.load_model("test_model", "1.0")

        # Verify
        assert loaded_model is not None
        assert model_id == "test_model_1.0"


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_minimal_pipeline(self, temp_dataset, temp_output):
        """Test minimal end-to-end pipeline."""
        # Create configuration
        config = TrainingConfig.quick_start(
            dataset_path=temp_dataset,
            architecture="resnet50", # <--- SUDAH BENAR
            num_classes=4,
            epochs=2,  # Short for testing
        )
        config.output_dir = temp_output
        config.model_config.weights = None  # No pretrained weights

        # Create pipeline
        pipeline = CoffeeBeanPipeline(config)

        # Run with single model
        results = pipeline.run(
            models_to_train=["resnet50"], # <--- FIX: Diubah dari "restnet50" (typo)
            auto_register=True,
            save_visualizations=False,  # Skip to save time
        )

        # Verify results
        assert "models_trained" in results
        assert len(results["models_trained"]) == 1
        assert "resnet50" in results["models_trained"] # <--- FIX: Diubah dari mobilenet_v3
        assert "best_model" in results
        assert results["best_model"] == "resnet50" # <--- FIX: Diubah dari mobilenet_v3


class TestComponentIntegration:
    """Test integration between specific components."""

    def test_data_to_model_integration(self, temp_dataset):
        """Test data pipeline to model integration."""
        # Setup data
        data_config = DataConfig(dataset_path=temp_dataset, batch_size=4)

        data_pipeline = CoffeeBeanDataPipeline(data_config)
        train_ds, _, _ = data_pipeline.load_dataset()

        # Create model
        model_config = ModelConfig(
            architecture="resnet50", num_classes=data_pipeline.get_num_classes(), weights=None # <--- FIX
        )

        model = ModelFactory.create("resnet50", model_config) # <--- FIX
        keras_model = model.build()

        # Try training for 1 step
        model.compile()

        # Get one batch
        for images, labels in train_ds.take(1):
            # Verify shapes match
            assert images.shape[1:] == (224, 224, 3)
            assert labels.shape[1] == 4

            # Try prediction
            predictions = keras_model.predict(images, verbose=0)
            assert predictions.shape == labels.shape

    def test_training_to_evaluation_integration(self, temp_dataset, temp_output):
        """Test training to evaluation integration."""
        # Quick training
        config = TrainingConfig.quick_start(
            dataset_path=temp_dataset, architecture="resnet50", epochs=1 # <--- FIX
        )
        config.output_dir = temp_output
        config.model_config.weights = None

        trainer = ModelTrainer(config)
        history = trainer.train_single_model("resnet50") # <--- FIX

        # Load trained model
        checkpoint_path = config.get_output_path("checkpoints") / "resnet50_best.h5" # <--- FIX

        if checkpoint_path.exists():
            model = tf.keras.models.load_model(checkpoint_path)

            # Evaluate
            _, _, test_ds = trainer.data_pipeline.get_datasets()

            evaluator = ClassificationEvaluator(model, trainer.data_pipeline.get_class_names())

            metrics = evaluator.evaluate(test_ds, verbose=0)

            assert "accuracy" in metrics
            assert metrics["accuracy"] >= 0.0
            assert metrics["accuracy"] <= 1.0


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_dataset_path(self):
        """Test handling of invalid dataset path."""
        with pytest.raises(Exception):
            config = DataConfig(dataset_path="/invalid/path")
            pipeline = CoffeeBeanDataPipeline(config)
            pipeline.load_dataset()

    def test_mismatched_config(self, temp_dataset):
        """Test handling of mismatched configurations."""
        # Create config with wrong number of classes
        data_config = DataConfig(dataset_path=temp_dataset)

        model_config = ModelConfig(
            architecture="resnet50", num_classes=10, weights=None  # Wrong! Should be 4
        )

        # This should work but produce wrong predictions
        # (not raise error, as it's a valid configuration)
        config = TrainingConfig(epochs=1, data_config=data_config, model_config=model_config)

        assert config.model_config.num_classes == 10