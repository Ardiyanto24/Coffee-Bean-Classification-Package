# Examples & Tutorials

## Example 1: Basic Training
```python
config = TrainingConfig.quick_start(
    dataset_path='/data',
    architecture='resnet50',
    epochs=50
)
trainer = ModelTrainer(config)
history = trainer.train_single_model('resnet50')
```

## Example 2: Custom Configuration
```python
config = TrainingConfig(
    epochs=100,
    batch_size=64,
    learning_rate=0.0001,
    data_config=DataConfig(
        dataset_path='/data',
        augmentation_params={'horizontal_flip': True}
    )
)
```

## Example 3: Model Comparison
```python
results = trainer.train_all_models(['resnet50', 'efficientnet_b0'])
best_model, score = trainer.get_best_model('val_accuracy')
print(f"Best: {best_model} - {score:.4f}")
```

## Example 4: Complete Pipeline
```python
pipeline = CoffeeBeanPipeline(config)
results = pipeline.run(
    models_to_train=['resnet50'],
    auto_register=True,
    save_visualizations=True
)
```

See `examples/` directory for complete runnable code.
