# User Guide - Coffee Bean Classification

Complete guide for using the package.

## Getting Started

### Installation
```bash
pip install -e .
```

### Quick Start
```python
from coffee_bean_classification import CoffeeBeanPipeline, TrainingConfig

config = TrainingConfig.quick_start(dataset_path='/data', architecture='resnet50')
pipeline = CoffeeBeanPipeline(config)
results = pipeline.run()
```

## Configuration

### YAML Configuration
```yaml
dataset_path: "/path/to/data"
architecture: "resnet50"
epochs: 50
batch_size: 32
learning_rate: 0.001
```

Load: `config = TrainingConfig.from_yaml('config.yaml')`

### Python Configuration
```python
config = TrainingConfig(
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)
```

## Training Workflows

### Single Model
```python
trainer = ModelTrainer(config)
history = trainer.train_single_model('resnet50')
```

### Multiple Models
```python
results = trainer.train_all_models(['resnet50', 'efficientnet_b0'])
```

### Resume Training
```python
history = trainer.resume_training('model.h5', additional_epochs=10)
```

## Evaluation

```python
evaluator = ClassificationEvaluator(model, class_names)
metrics = evaluator.evaluate(test_dataset)
evaluator.plot_confusion_matrix()
evaluator.generate_report()
```

## Model Registry

```python
registry = ModelRegistry()
registry.register_model(model, metadata)
loaded = registry.load_model('resnet50', version='latest')
```

See API Reference for complete details.
