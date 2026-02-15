"""
Example usage of the Data Pipeline module.

This script demonstrates how to use the data pipeline for loading,
preprocessing, and augmenting coffee bean images.
"""

from pathlib import Path
from coffee_bean_classification.configs import DataConfig
from coffee_bean_classification.data import (
    CoffeeBeanDataPipeline,
    DataAugmentation
)


def example_1_basic_pipeline():
    """Example 1: Basic data pipeline usage."""
    print("=" * 60)
    print("Example 1: Basic Data Pipeline")
    print("=" * 60)
    
    # Create data configuration
    config = DataConfig(
        dataset_path='/path/to/coffee_beans_dataset',
        image_size=(224, 224),
        batch_size=32,
        split_ratio=(0.7, 0.15, 0.15)
    )
    
    # Create pipeline
    pipeline = CoffeeBeanDataPipeline(config)
    
    # Load datasets
    train_ds, val_ds, test_ds = pipeline.load_dataset()
    
    print(f"✓ Training batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f"✓ Validation batches: {tf.data.experimental.cardinality(val_ds)}")
    print(f"✓ Test batches: {tf.data.experimental.cardinality(test_ds)}")
    print(f"✓ Classes: {pipeline.get_class_names()}")
    print()


def example_2_augmentation_strategies():
    """Example 2: Different augmentation strategies."""
    print("=" * 60)
    print("Example 2: Augmentation Strategies")
    print("=" * 60)
    
    # Try different strategies
    strategies = ['none', 'light', 'medium', 'heavy']
    
    for strategy in strategies:
        aug = DataAugmentation(strategy=strategy)
        print(f"✓ {strategy.capitalize()} augmentation:")
        print(f"  Config: {aug.config}")
    
    print()


def example_3_custom_augmentation():
    """Example 3: Custom augmentation configuration."""
    print("=" * 60)
    print("Example 3: Custom Augmentation")
    print("=" * 60)
    
    # Create custom augmentation
    custom_config = {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotation_range': 0.25,
        'zoom_range': 0.15,
        'brightness_range': (0.7, 1.3),
        'contrast_range': (0.7, 1.3)
    }
    
    config = DataConfig(
        dataset_path='/path/to/dataset',
        augmentation_params=custom_config
    )
    
    pipeline = CoffeeBeanDataPipeline(config)
    
    print("✓ Custom augmentation configured")
    print(f"  Parameters: {custom_config}")
    print()


def example_4_dataset_info():
    """Example 4: Getting dataset information."""
    print("=" * 60)
    print("Example 4: Dataset Information")
    print("=" * 60)
    
    config = DataConfig(dataset_path='/path/to/dataset')
    pipeline = CoffeeBeanDataPipeline(config)
    pipeline.load_dataset()
    
    # Get dataset info
    info = pipeline.get_dataset_info()
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get class distribution
    distribution = pipeline.get_class_distribution()
    print("\nClass Distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count} images")
    
    print()


def example_5_visualization():
    """Example 5: Visualizing samples."""
    print("=" * 60)
    print("Example 5: Visualization")
    print("=" * 60)
    
    config = DataConfig(dataset_path='/path/to/dataset')
    pipeline = CoffeeBeanDataPipeline(config)
    pipeline.load_dataset()
    
    # Visualize training samples
    print("Displaying 9 random training samples...")
    pipeline.visualize_samples(n=9, dataset='train')
    
    # Show augmentation examples
    print("Displaying augmentation examples...")
    pipeline.show_augmentation_examples(image_idx=0, n_examples=6)
    
    print()


def example_6_batch_processing():
    """Example 6: Processing batches."""
    print("=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)
    
    config = DataConfig(
        dataset_path='/path/to/dataset',
        batch_size=32
    )
    pipeline = CoffeeBeanDataPipeline(config)
    train_ds, val_ds, test_ds = pipeline.load_dataset()
    
    # Get a sample batch
    images, labels = pipeline.get_sample_batch('train')
    
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
    
    # Process batch
    print("\nProcessing batch...")
    for batch_images, batch_labels in train_ds.take(1):
        print(f"  Processed {batch_images.shape[0]} images")
        print(f"  Classes in batch: {tf.reduce_sum(batch_labels, axis=0)}")
    
    print()


def example_7_performance_optimization():
    """Example 7: Performance optimization."""
    print("=" * 60)
    print("Example 7: Performance Optimization")
    print("=" * 60)
    
    config = DataConfig(
        dataset_path='/path/to/dataset',
        batch_size=32,
        cache_dir='/tmp/coffee_cache',  # Enable caching
        num_parallel_calls=-1,  # Use AUTOTUNE
        prefetch_buffer_size=-1  # Use AUTOTUNE
    )
    
    pipeline = CoffeeBeanDataPipeline(config)
    train_ds, val_ds, test_ds = pipeline.load_dataset()
    
    print("✓ Dataset optimized with:")
    print("  - Caching enabled")
    print("  - Parallel data loading (AUTOTUNE)")
    print("  - Prefetching (AUTOTUNE)")
    
    # Benchmark
    import time
    print("\nBenchmarking...")
    
    start = time.time()
    for _ in train_ds.take(10):
        pass
    first_10_batches = time.time() - start
    
    start = time.time()
    for _ in train_ds.take(10):
        pass
    second_10_batches = time.time() - start
    
    print(f"  First 10 batches: {first_10_batches:.2f}s")
    print(f"  Second 10 batches: {second_10_batches:.2f}s (cached)")
    print(f"  Speedup: {first_10_batches/second_10_batches:.2f}x")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Coffee Bean Classification - Data Pipeline Examples    ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Note: These examples assume you have a dataset
    # Replace '/path/to/dataset' with your actual dataset path
    
    print("📌 Note: Update dataset paths before running examples")
    print()
    
    try:
        example_1_basic_pipeline()
        example_2_augmentation_strategies()
        example_3_custom_augmentation()
        example_4_dataset_info()
        # example_5_visualization()  # Uncomment if you want to see plots
        example_6_batch_processing()
        example_7_performance_optimization()
        
        print("=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure to:")
        print("1. Update dataset paths")
        print("2. Have a properly structured dataset")
        print("3. Install all dependencies")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import tensorflow as tf
    main()
