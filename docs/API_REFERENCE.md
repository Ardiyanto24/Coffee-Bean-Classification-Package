# API Reference

Complete API documentation for Coffee Bean Classification package.

## Table of Contents
1. [configs](#configs-module)
2. [data](#data-module)
3. [models](#models-module)
4. [training](#training-module)
5. [evaluation](#evaluation-module)
6. [registry](#registry-module)
7. [pipeline](#pipeline-module)

---

## configs Module

Configuration management classes.

### TrainingConfig
Main training configuration.

**Signature:**
```python
TrainingConfig(
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    optimizer: str = 'adam',
    loss: str = 'categorical_crossentropy',
    metrics: List[str] = ['accuracy'],
    seed: int = 42,
    data_config: Optional[DataConfig] = None,
    model_config: Optional[ModelConfig] = None
)
```

**Methods:**
- `validate() -> bool`: Validate configuration
- `save(path: str) -> Path`: Save to YAML/JSON
- `load(path: str) -> TrainingConfig`: Load from file
- `quick_start(dataset_path, architecture, **kwargs) -> TrainingConfig`: Quick configuration

**Example:**
```python
config = TrainingConfig.quick_start(
    dataset_path='/data',
    architecture='resnet50',
    epochs=50
)
```

### DataConfig
Data pipeline configuration.

**Signature:**
```python
DataConfig(
    dataset_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
)
```

### ModelConfig
Model architecture configuration.

**Signature:**
```python
ModelConfig(
    architecture: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    weights: Optional[str] = 'imagenet'
)
```

---

## data Module

Data loading and augmentation.

### CoffeeBeanDataPipeline
Main data pipeline.

**Signature:**
```python
CoffeeBeanDataPipeline(config: DataConfig)
```

**Methods:**
- `load_dataset() -> Tuple[Dataset, Dataset, Dataset]`: Load train/val/test
- `visualize_samples(n: int, save_path: str)`: Show samples
- `get_class_distribution() -> Dict[str, int]`: Get class counts

**Example:**
```python
pipeline = CoffeeBeanDataPipeline(config)
train_ds, val_ds, test_ds = pipeline.load_dataset()
```

### DataAugmentation
Image augmentation strategies.

**Strategies:** none, light, medium, heavy

---

## models Module

Model factory and architectures.

### ModelFactory
Factory for creating models.

**Methods:**
- `create(model_name: str, config: ModelConfig) -> BaseModel`
- `list_available() -> List[str]`
- `register(name: str)`: Decorator for custom models

**Available Models:**
- resnet50
- efficientnet_b0
- efficientnet_b3
- mobilenet_v3
- densenet121

**Example:**
```python
model = ModelFactory.create('resnet50', config)
keras_model = model.build()
```

---

## training Module

Training orchestration.

### ModelTrainer
Main trainer class.

**Signature:**
```python
ModelTrainer(
    config: TrainingConfig,
    data_pipeline: Optional[CoffeeBeanDataPipeline] = None
)
```

**Methods:**
- `train_single_model(model_name: str) -> History`
- `train_all_models(models: List[str]) -> Dict`
- `resume_training(model_path: str, epochs: int) -> History`
- `get_best_model(metric: str) -> Tuple[str, float]`

**Example:**
```python
trainer = ModelTrainer(config)
history = trainer.train_single_model('resnet50')
```

---

## evaluation Module

Model evaluation and comparison.

### ClassificationEvaluator
Evaluate single model.

**Signature:**
```python
ClassificationEvaluator(
    model: tf.keras.Model,
    class_names: List[str]
)
```

**Methods:**
- `evaluate(dataset: Dataset) -> Dict[str, Any]`
- `plot_confusion_matrix(save_path: str)`
- `plot_roc_curves(save_path: str)`
- `generate_report(save_path: str) -> str`

**Available Metrics:**
- accuracy, balanced_accuracy
- precision_macro, precision_weighted
- recall_macro, recall_weighted
- f1_macro, f1_weighted
- roc_auc_macro, roc_auc_weighted
- top_2_accuracy, top_3_accuracy

### ModelComparator
Compare multiple models.

**Methods:**
- `add_model(name: str, metrics: Dict)`
- `get_comparison_df() -> DataFrame`
- `rank_models(metric: str) -> DataFrame`
- `get_best_model(metric: str) -> Tuple[str, float]`

---

## registry Module

Model version control.

### ModelRegistry
Manage model versions.

**Signature:**
```python
ModelRegistry(registry_path: str = 'model_registry')
```

**Methods:**
- `register_model(model: Model, metadata: ModelMetadata) -> str`
- `load_model(model_name: str, version: str) -> Model`
- `get_best_model(metric: str) -> str`
- `list_models() -> List[str]`

---

## pipeline Module

End-to-end workflow.

### CoffeeBeanPipeline
Complete pipeline.

**Signature:**
```python
CoffeeBeanPipeline(config: TrainingConfig)
```

**Methods:**
- `run(models: List[str], auto_register: bool, save_viz: bool) -> Dict`
- `get_best_model() -> Model`

**Example:**
```python
pipeline = CoffeeBeanPipeline(config)
results = pipeline.run()
best_model = pipeline.get_best_model()
```
