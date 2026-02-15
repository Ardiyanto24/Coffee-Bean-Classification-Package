"""
Example usage of the Coffee Bean Classification package.

This script demonstrates how to use the configuration system
and utilities from Phase 2.
"""

from pathlib import Path
from coffee_bean_classification.configs import (
    DataConfig,
    ModelConfig,
    TrainingConfig
)
from coffee_bean_classification.utils import (
    set_global_seed,
    FileManager,
    setup_logger,
    verify_reproducibility
)


def example_1_basic_config():
    """Example 1: Create basic configuration programmatically."""
    print("=" * 60)
    print("Example 1: Basic Configuration")
    print("=" * 60)
    
    # Create configuration with quick_start
    config = TrainingConfig.quick_start(
        dataset_path='/path/to/dataset',
        architecture='resnet50',
        num_classes=4,
        epochs=50
    )
    
    # Validate
    config.validate()
    
    print(config)
    print("\n✓ Configuration created and validated")
    print()


def example_2_manual_config():
    """Example 2: Create configuration manually with all components."""
    print("=" * 60)
    print("Example 2: Manual Configuration")
    print("=" * 60)
    
    # Step 1: Configure data pipeline
    data_config = DataConfig(
        dataset_path='/path/to/dataset',
        image_size=(224, 224),
        batch_size=32,
        split_ratio=(0.7, 0.15, 0.15)
    )
    print("✓ DataConfig created")
    
    # Step 2: Configure model
    model_config = ModelConfig.for_architecture(
        'efficientnet_b0',
        num_classes=4
    )
    print("✓ ModelConfig created")
    
    # Step 3: Configure training
    training_config = TrainingConfig(
        epochs=50,
        learning_rate=0.001,
        data_config=data_config,
        model_config=model_config,
        experiment_name='my_experiment'
    )
    print("✓ TrainingConfig created")
    
    # Validate all
    training_config.validate()
    print("\n✓ All configurations validated")
    print()


def example_3_save_load_config():
    """Example 3: Save and load configurations."""
    print("=" * 60)
    print("Example 3: Save and Load Configuration")
    print("=" * 60)
    
    # Create config
    config = TrainingConfig.quick_start(
        dataset_path='/path/to/dataset',
        architecture='resnet50',
        num_classes=4
    )
    
    # Save as YAML
    yaml_path = Path('temp_config.yaml')
    config.save(yaml_path)
    print(f"✓ Configuration saved to {yaml_path}")
    
    # Load from YAML
    loaded_config = TrainingConfig.load(yaml_path)
    print(f"✓ Configuration loaded from {yaml_path}")
    
    # Verify they're the same
    assert config == loaded_config
    print("✓ Loaded config matches original")
    
    # Cleanup
    yaml_path.unlink()
    print()


def example_4_utilities():
    """Example 4: Use utility functions."""
    print("=" * 60)
    print("Example 4: Utility Functions")
    print("=" * 60)
    
    # Setup logger
    logger = setup_logger('example', level='INFO')
    logger.info('Logger initialized')
    print("✓ Logger configured")
    
    # Set seed for reproducibility
    set_global_seed(42)
    print("✓ Global seed set to 42")
    
    # Verify reproducibility
    is_reproducible = verify_reproducibility(42, n_trials=3)
    print(f"✓ Reproducibility verified: {is_reproducible}")
    
    # Setup file manager
    file_manager = FileManager(
        base_dir='experiments',
        experiment_name='test_experiment'
    )
    print(f"✓ FileManager created at {file_manager.run_dir}")
    
    # Get organized paths
    model_path = file_manager.get_model_path('resnet50', 'best')
    log_path = file_manager.get_log_path('training')
    
    print(f"  Model path: {model_path}")
    print(f"  Log path: {log_path}")
    print()


def example_5_config_updates():
    """Example 5: Update and merge configurations."""
    print("=" * 60)
    print("Example 5: Update and Merge Configurations")
    print("=" * 60)
    
    # Create base config
    config = TrainingConfig(epochs=50, batch_size=32)
    print("Base config:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    
    # Update config
    config.update(epochs=100, learning_rate=0.0001)
    print("\nAfter update:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Create a copy
    config_copy = config.copy()
    config_copy.update(epochs=200)
    print("\nCopy with different epochs:")
    print(f"  Original epochs: {config.epochs}")
    print(f"  Copy epochs: {config_copy.epochs}")
    
    print("\n✓ Config update and copy operations successful")
    print()


def example_6_model_configs():
    """Example 6: Different model configurations."""
    print("=" * 60)
    print("Example 6: Model Configurations")
    print("=" * 60)
    
    # Test all supported architectures
    architectures = ['resnet50', 'efficientnet_b0', 'efficientnet_b3', 
                    'mobilenet_v3', 'densenet121']
    
    for arch in architectures:
        config = ModelConfig.for_architecture(arch, num_classes=4)
        config.validate()
        
        print(f"✓ {arch}:")
        print(f"  Input size: {config.get_input_size()}")
        print(f"  Dropout: {config.dropout_rate}")
        print(f"  Transfer learning: {config.is_transfer_learning()}")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Coffee Bean Classification - Phase 2 Examples          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    try:
        example_1_basic_config()
        example_2_manual_config()
        example_3_save_load_config()
        example_4_utilities()
        example_5_config_updates()
        example_6_model_configs()
        
        print("=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
