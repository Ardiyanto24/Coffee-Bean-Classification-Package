"""
Example usage of the Training System.

This script demonstrates how to use the training system for training
coffee bean classification models.
"""

from coffee_bean_classification.configs import TrainingConfig
from coffee_bean_classification.training import ModelTrainer


def example_1_basic_training():
    """Example 1: Basic single model training."""
    print("=" * 60)
    print("Example 1: Basic Training")
    print("=" * 60)
    
    # Create configuration using quick_start
    config = TrainingConfig.quick_start(
        dataset_path='/path/to/coffee_beans',
        architecture='resnet50',
        num_classes=4,
        epochs=50
    )
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Train single model
    history = trainer.train_single_model('resnet50')
    
    print(f"✓ Training completed")
    print(f"✓ Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"✓ Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    print()


def example_2_multi_model_training():
    """Example 2: Train multiple models."""
    print("=" * 60)
    print("Example 2: Multi-Model Training")
    print("=" * 60)
    
    config = TrainingConfig.quick_start(
        dataset_path='/path/to/coffee_beans',
        num_classes=4,
        epochs=30
    )
    
    trainer = ModelTrainer(config)
    
    # Train all 5 models
    models_to_train = [
        'resnet50',
        'efficientnet_b0',
        'efficientnet_b3',
        'mobilenet_v3',
        'densenet121'
    ]
    
    results = trainer.train_all_models(models_to_train)
    
    # Compare results
    print("\nResults:")
    for model_name, history in results.items():
        best_acc = max(history.history['val_accuracy'])
        print(f"  {model_name}: {best_acc:.4f}")
    
    # Get best model
    best_model, best_value = trainer.get_best_model('val_accuracy', 'max')
    print(f"\n✓ Best model: {best_model} ({best_value:.4f})")
    print()


def example_3_custom_callbacks():
    """Example 3: Training with custom callbacks."""
    print("=" * 60)
    print("Example 3: Custom Callbacks")
    print("=" * 60)
    
    import tensorflow as tf
    
    config = TrainingConfig.quick_start(
        dataset_path='/path/to/coffee_beans',
        architecture='mobilenet_v3',
        epochs=20
    )
    
    # Create custom callback
    def on_epoch_end(epoch, logs):
        acc = logs.get('val_accuracy', 0)
        if acc > 0.95:
            print(f"\n✨ Reached 95% accuracy at epoch {epoch}!")
    
    custom_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=on_epoch_end
    )
    
    trainer = ModelTrainer(config)
    history = trainer.train_single_model(
        'mobilenet_v3',
        custom_callbacks=[custom_callback]
    )
    
    print("✓ Training with custom callback completed")
    print()


def example_4_resume_training():
    """Example 4: Resume training from checkpoint."""
    print("=" * 60)
    print("Example 4: Resume Training")
    print("=" * 60)
    
    config = TrainingConfig.quick_start(
        dataset_path='/path/to/coffee_beans',
        architecture='efficientnet_b0',
        epochs=20
    )
    
    trainer = ModelTrainer(config)
    
    # First training session
    print("Initial training...")
    trainer.train_single_model('efficientnet_b0')
    
    # Resume training for 10 more epochs
    print("\nResuming training...")
    model_path = config.get_output_path('checkpoints') / 'efficientnet_b0_best.h5'
    
    history = trainer.resume_training(
        model_path=str(model_path),
        additional_epochs=10,
        model_name='efficientnet_b0_resumed'
    )
    
    print("✓ Resumed training completed")
    print()


def example_5_custom_configuration():
    """Example 5: Fully customized training configuration."""
    print("=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    from coffee_bean_classification.configs import DataConfig, ModelConfig
    
    # Custom data configuration
    data_config = DataConfig(
        dataset_path='/path/to/coffee_beans',
        image_size=(300, 300),  # Larger images
        batch_size=16,
        split_ratio=(0.8, 0.1, 0.1),
        augmentation_params={
            'horizontal_flip': True,
            'rotation_range': 0.3,
            'zoom_range': 0.15,
            'brightness_range': (0.7, 1.3)
        }
    )
    
    # Custom model configuration
    model_config = ModelConfig(
        architecture='densenet121',
        input_shape=(300, 300, 3),
        num_classes=4,
        weights='imagenet',
        dropout_rate=0.4,
        freeze_backbone=True
    )
    
    # Custom training configuration
    training_config = TrainingConfig(
        epochs=100,
        learning_rate=0.0001,
        optimizer='adamw',
        data_config=data_config,
        model_config=model_config,
        early_stopping_patience=15,
        reduce_lr_patience=7,
        reduce_lr_factor=0.3,
        experiment_name='densenet_custom'
    )
    
    # Enable TensorBoard
    training_config.enable_callback('tensorboard')
    
    trainer = ModelTrainer(training_config)
    history = trainer.train_single_model('densenet121')
    
    print("✓ Custom configuration training completed")
    print()


def example_6_learning_rate_scheduling():
    """Example 6: Training with learning rate scheduling."""
    print("=" * 60)
    print("Example 6: Learning Rate Scheduling")
    print("=" * 60)
    
    config = TrainingConfig.quick_start(
        dataset_path='/path/to/coffee_beans',
        architecture='resnet50',
        epochs=50
    )
    
    trainer = ModelTrainer(config)
    
    # Create LR scheduler
    lr_scheduler = trainer.callback_manager.create_learning_rate_scheduler(
        schedule_type='cosine',
        initial_lr=0.001,
        total_epochs=50
    )
    
    # Train with LR scheduler
    history = trainer.train_single_model(
        'resnet50',
        custom_callbacks=[lr_scheduler]
    )
    
    print("✓ Training with LR scheduling completed")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Coffee Bean Classification - Training Examples         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    print("📌 Note: Update dataset paths before running examples")
    print()
    
    try:
        # Uncomment to run specific examples
        # example_1_basic_training()
        # example_2_multi_model_training()
        # example_3_custom_callbacks()
        # example_4_resume_training()
        # example_5_custom_configuration()
        # example_6_learning_rate_scheduling()
        
        print("=" * 60)
        print("✓ Examples ready to run!")
        print("=" * 60)
        print("\nUncomment the examples you want to run in main()")
        print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
