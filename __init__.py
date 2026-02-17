"""
Coffee Bean Classification Package
===================================

A production-ready, OOP-based package for coffee bean image classification.

Features:
---------
- Modular architecture with 7 core components
- Support for multiple CNN architectures (ResNet, EfficientNet, MobileNet, DenseNet)
- Flexible configuration system (YAML/JSON)
- Built-in data augmentation strategies
- Model versioning and registry
- Experiment tracking (MLflow/WandB)
- Comprehensive evaluation and comparison tools

Quick Start:
-----------
>>> from coffee_bean_classification import TrainingConfig, ModelTrainer
>>> config = TrainingConfig.from_yaml('config.yaml')
>>> trainer = ModelTrainer(config)
>>> results = trainer.train_all_models(['resnet50', 'efficientnet_b0'])

Components:
----------
- configs: Configuration management
- data: Data pipeline and augmentation
- models: Model factory and architectures
- training: Training orchestration
- evaluation: Metrics and model comparison
- registry: Model versioning and storage
- experiments: Experiment tracking
- utils: Utility functions

Author: Coffee Bean Classification Team
License: MIT
Version: 0.1.0
"""

from .version import __version__, __author__, __description__

# Core imports will be added as we implement each component
# from .configs import TrainingConfig, DataConfig, ModelConfig
# from .data import CoffeeBeanDataPipeline
# from .models import ModelFactory
# from .training import ModelTrainer
# from .evaluation import ClassificationEvaluator, ModelComparator
# from .registry import ModelRegistry
# from .experiments import ExperimentManager

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    # Components will be added progressively
]

# Package metadata
__title__ = "coffee_bean_classification"
__url__ = "https://github.com/yourusername/coffee_bean_classification"
__status__ = "Development"
