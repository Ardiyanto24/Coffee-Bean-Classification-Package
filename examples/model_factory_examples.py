"""
Example usage of the Model Factory module.

This script demonstrates how to use the model factory for creating
and working with different CNN architectures.
"""

from coffee_bean_classification.configs import ModelConfig
from coffee_bean_classification.models import ModelFactory


def example_1_basic_model_creation():
    """Example 1: Basic model creation."""
    print("=" * 60)
    print("Example 1: Basic Model Creation")
    print("=" * 60)

    # Create configuration
    config = ModelConfig.for_architecture("resnet50", num_classes=4)

    # Create model using factory
    model = ModelFactory.create("resnet50", config)

    # Build the model
    keras_model = model.build()

    print(f"✓ Model created: {keras_model.name}")
    print(f"✓ Input shape: {keras_model.input_shape}")
    print(f"✓ Output shape: {keras_model.output_shape}")

    # Count parameters
    params = model.count_parameters()
    print(f"✓ Total parameters: {params['total']:,}")
    print(f"✓ Trainable parameters: {params['trainable']:,}")
    print()


def example_2_all_architectures():
    """Example 2: Create all available architectures."""
    print("=" * 60)
    print("Example 2: All Available Architectures")
    print("=" * 60)

    # List available models
    available = ModelFactory.list_available()
    print(f"Available models: {available}\n")

    # Create all models
    for model_name in available:
        config = ModelConfig.for_architecture(model_name, num_classes=4)
        model = ModelFactory.create(model_name, config)
        model.build()
        params = model.count_parameters()

        print(f"✓ {model_name.upper()}:")
        print(f"  Input size: {config.get_input_size()}")
        print(f"  Parameters: {params['total']:,}")

    print()


def example_3_transfer_learning():
    """Example 3: Transfer learning setup."""
    print("=" * 60)
    print("Example 3: Transfer Learning")
    print("=" * 60)

    # Create config with pretrained weights
    config = ModelConfig(
        architecture="efficientnet_b0",
        input_shape=(224, 224, 3),
        num_classes=4,
        weights="imagenet",  # Use pretrained weights
        freeze_backbone=True,  # Freeze for transfer learning
        dropout_rate=0.3,
    )

    # Create and build model
    model = ModelFactory.create("efficientnet_b0", config)
    model.build()

    params = model.count_parameters()
    print("Transfer Learning Setup:")
    print("✓ Using ImageNet pretrained weights")
    print(f"✓ Backbone frozen: {not model.backbone.trainable}")
    print(f"✓ Total params: {params['total']:,}")
    print(f"✓ Trainable params: {params['trainable']:,}")
    print(f"✓ Non-trainable params: {params['non_trainable']:,}")

    # Later, for fine-tuning
    print("\nFine-tuning (unfreeze last 20 layers):")
    model.unfreeze_backbone(from_layer=-20)
    params_finetuned = model.count_parameters()
    print(f"✓ Trainable params: {params_finetuned['trainable']:,}")

    print()


def example_4_model_comparison():
    """Example 4: Compare models."""
    print("=" * 60)
    print("Example 4: Model Comparison")
    print("=" * 60)

    comparison = ModelFactory.compare_models()

    print(f"{'Model':<20} {'Params':<15} {'Speed':<15} {'Accuracy':<15}")
    print("-" * 65)

    for name, info in comparison.items():
        params = info.get("approximate_params", "N/A")
        if isinstance(params, int):
            params = f"{params:,}"
        speed = info.get("relative_speed", "N/A")
        accuracy = info.get("relative_accuracy", "N/A")

        print(f"{name:<20} {params:<15} {speed:<15} {accuracy:<15}")

    print()


def example_5_custom_model_registration():
    """Example 5: Register custom model."""
    print("=" * 60)
    print("Example 5: Custom Model Registration")
    print("=" * 60)

    # Define custom model
    import tensorflow as tf

    from coffee_bean_classification.models import BaseModel

    @ModelFactory.register("simple_cnn")
    class SimpleCNN(BaseModel):
        """Simple CNN for demonstration."""

        def build(self):
            inputs = tf.keras.Input(shape=self.config.input_shape)

            x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
            outputs = tf.keras.layers.Dense(self.config.num_classes, activation="softmax")(x)

            self.model = tf.keras.Model(inputs, outputs, name="SimpleCNN")
            return self.model

    # Use custom model
    config = ModelConfig(architecture="simple_cnn", input_shape=(224, 224, 3), num_classes=4)

    model = ModelFactory.create("simple_cnn", config)
    keras_model = model.build()

    print("✓ Custom model registered and created")
    print(f"✓ Model name: {keras_model.name}")
    print(f"✓ Available models: {ModelFactory.list_available()}")

    # Cleanup
    ModelFactory.unregister("simple_cnn")
    print()


def example_6_model_compilation():
    """Example 6: Compile models."""
    print("=" * 60)
    print("Example 6: Model Compilation")
    print("=" * 60)

    config = ModelConfig.for_architecture("mobilenet_v3", num_classes=4)
    model = ModelFactory.create("mobilenet_v3", config)

    # Compile with custom settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy"],
    )

    print("✓ Model compiled with:")
    print("  - Optimizer: Adam (lr=0.001)")
    print("  - Loss: categorical_crossentropy")
    print("  - Metrics: accuracy, top_k_categorical_accuracy")

    # Get the Keras model
    keras_model = model.get_model()
    keras_model.summary()

    print()


def example_7_model_info():
    """Example 7: Get model information."""
    print("=" * 60)
    print("Example 7: Model Information")
    print("=" * 60)

    # Get info for each model
    for model_name in ModelFactory.list_available():
        info = ModelFactory.get_model_info(model_name)

        print(f"\n{model_name.upper()}:")
        print(f"  Class: {info['class_name']}")
        print(f"  Module: {info['module']}")
        if info["docstring"]:
            # Print first line of docstring
            first_line = info["docstring"].strip().split("\n")[0]
            print(f"  Description: {first_line}")

    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Coffee Bean Classification - Model Factory Examples    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    try:
        example_1_basic_model_creation()
        example_2_all_architectures()
        example_3_transfer_learning()
        example_4_model_comparison()
        example_5_custom_model_registration()
        example_6_model_compilation()
        example_7_model_info()

        print("=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import tensorflow as tf

    # Suppress TF warnings for cleaner output
    tf.get_logger().setLevel("ERROR")
    main()
