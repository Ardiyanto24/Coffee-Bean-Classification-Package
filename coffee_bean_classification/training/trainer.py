"""Model trainer for orchestrating training process."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from ..data import CoffeeBeanDataPipeline
from ..models import ModelFactory
from ..utils import get_logger, save_json, set_global_seed
from .callbacks import CallbackManager, ProgressCallback

logger = get_logger(__name__)


class ModelTrainer:
    """
    Orchestrates model training process.

    Handles:
    - Model creation
    - Training execution
    - Callback management
    - History tracking
    - Multi-model training

    Example:
        >>> from coffee_bean_classification.configs import TrainingConfig
        >>> config = TrainingConfig.quick_start(...)
        >>> trainer = ModelTrainer(config)
        >>> history = trainer.train_single_model('resnet50')
        >>> results = trainer.train_all_models(['resnet50', 'efficientnet_b0'])
    """

    def __init__(self, config, data_pipeline: Optional[CoffeeBeanDataPipeline] = None):
        """
        Initialize model trainer.

        Args:
            config: TrainingConfig instance
            data_pipeline: Optional pre-configured data pipeline
        """
        self.config = config

        # Validate config
        self.config.validate()

        # Set seed for reproducibility
        set_global_seed(self.config.seed)
        logger.info(f"Set random seed to {self.config.seed}")

        # Initialize data pipeline
        if data_pipeline is None:
            if self.config.data_config is None:
                raise ValueError("Either provide data_pipeline or set config.data_config")
            self.data_pipeline = CoffeeBeanDataPipeline(self.config.data_config)
        else:
            self.data_pipeline = data_pipeline

        # Initialize callback manager
        self.callback_manager = CallbackManager(config)

        # Training history storage
        self.training_history: Dict[str, Any] = {}

        logger.info("ModelTrainer initialized")

    def train_single_model(
        self,
        model_name: str,
        custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        verbose: Optional[int] = None,
    ) -> tf.keras.callbacks.History:
        """
        Train a single model.

        Args:
            model_name: Name of the model architecture
            custom_callbacks: Additional custom callbacks
            verbose: Verbosity level (0, 1, or 2)

        Returns:
            Training history

        Example:
            >>> history = trainer.train_single_model('resnet50')
            >>> print(f"Best accuracy: {max(history.history['val_accuracy'])}")
        """
        logger.info("=" * 60)
        logger.info(f"Training {model_name}")
        logger.info("=" * 60)

        # Load datasets if needed
        train_ds, val_ds, test_ds = self.data_pipeline.get_datasets(load_if_needed=True)

        # Create model
        if self.config.model_config is None:
            from ..configs import ModelConfig

            model_config = ModelConfig.for_architecture(
                model_name, num_classes=self.data_pipeline.get_num_classes()
            )
        else:
            model_config = self.config.model_config

        logger.info(f"Creating {model_name} model...")
        model = ModelFactory.create(model_name, model_config)

        # Build model
        keras_model = model.build()

        # Log model info
        params = model.count_parameters()
        logger.info(
            f"Model built: {params['total']:,} total params " f"({params['trainable']:,} trainable)"
        )

        # Compile model
        logger.info("Compiling model...")
        model.compile(
            optimizer=self._get_optimizer(), loss=self.config.loss, metrics=self.config.metrics
        )

        # Get callbacks
        all_callbacks = self.callback_manager.get_callbacks(
            model_name=model_name, custom_callbacks=custom_callbacks
        )

        # Add progress callback
        all_callbacks.append(ProgressCallback(self.config.epochs, model_name))

        # Train model
        logger.info(f"Starting training for {self.config.epochs} epochs...")

        if verbose is None:
            verbose = self.config.verbose

        history = keras_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=all_callbacks,
            verbose=verbose,
        )

        # Save history
        self._save_training_history(model_name, history)

        # Store in memory
        self.training_history[model_name] = {
            "history": history.history,
            "params": params,
            "config": model_config.to_dict(),
        }

        logger.info(f"✓ Training completed for {model_name}")
        logger.info("=" * 60)

        return history

    def train_all_models(
        self, model_list: Optional[List[str]] = None, continue_on_error: bool = True
    ) -> Dict[str, tf.keras.callbacks.History]:
        """
        Train multiple models sequentially.

        Args:
            model_list: List of model names to train (None for all available)
            continue_on_error: If True, continue training other models if one fails

        Returns:
            Dictionary mapping model names to training histories

        Example:
            >>> models = ['resnet50', 'efficientnet_b0', 'mobilenet_v3']
            >>> results = trainer.train_all_models(models)
            >>> for name, history in results.items():
            >>>     print(f"{name}: {max(history.history['val_accuracy']):.4f}")
        """
        if model_list is None:
            model_list = ModelFactory.list_available()

        logger.info(f"Training {len(model_list)} models: {model_list}")

        results = {}
        failed_models = []

        for i, model_name in enumerate(model_list, 1):
            logger.info(f"\n{'#' * 60}")
            logger.info(f"Model {i}/{len(model_list)}: {model_name}")
            logger.info(f"{'#' * 60}\n")

            try:
                history = self.train_single_model(model_name)
                results[model_name] = history
                logger.info(f"✓ {model_name} training successful")

            except Exception as e:
                logger.error(f"✗ Failed to train {model_name}: {str(e)}")
                failed_models.append(model_name)

                if not continue_on_error:
                    raise

        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("Training Summary")
        logger.info(f"{'=' * 60}")
        logger.info(f"Successful: {len(results)}/{len(model_list)}")
        if failed_models:
            logger.warning(f"Failed: {failed_models}")
        logger.info(f"{'=' * 60}\n")

        # Save overall summary
        self._save_training_summary(results, failed_models)

        return results

    def resume_training(
        self, model_path: str, additional_epochs: int, model_name: str = "resumed_model"
    ) -> tf.keras.callbacks.History:
        """
        Resume training from a saved model.

        Args:
            model_path: Path to saved model
            additional_epochs: Number of additional epochs to train
            model_name: Name for the resumed training session

        Returns:
            Training history for additional epochs
        """
        logger.info(f"Resuming training from {model_path}")

        # Load model
        keras_model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")

        # Get datasets
        train_ds, val_ds, _ = self.data_pipeline.get_datasets(load_if_needed=True)

        # Get callbacks
        callbacks = self.callback_manager.get_callbacks(model_name=model_name)
        callbacks.append(ProgressCallback(additional_epochs, model_name))

        # Continue training
        history = keras_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=additional_epochs,
            callbacks=callbacks,
            verbose=self.config.verbose,
        )

        logger.info("✓ Resumed training completed")
        return history

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Get optimizer instance.

        Returns:
            Keras optimizer
        """
        optimizer_name = self.config.optimizer.lower()
        lr = self.config.learning_rate

        if optimizer_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif optimizer_name == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer_name == "adamw":
            return tf.keras.optimizers.AdamW(learning_rate=lr)
        else:
            logger.warning(f"Unknown optimizer '{optimizer_name}', using Adam")
            return tf.keras.optimizers.Adam(learning_rate=lr)

    def _save_training_history(self, model_name: str, history: tf.keras.callbacks.History) -> None:
        """Save training history to JSON."""
        history_dict = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "optimizer": self.config.optimizer,
            },
            "history": history.history,
        }

        # Convert numpy values to float
        for key in history_dict["history"]:
            history_dict["history"][key] = [float(v) for v in history_dict["history"][key]]

        filepath = self.config.get_output_path("histories") / f"{model_name}_history.json"
        save_json(history_dict, filepath)

        logger.debug(f"Saved training history to {filepath}")

    def _save_training_summary(
        self, results: Dict[str, tf.keras.callbacks.History], failed_models: List[str]
    ) -> None:
        """Save overall training summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(results) + len(failed_models),
            "successful_models": len(results),
            "failed_models": failed_models,
            "models": {},
        }

        # Add summary for each successful model
        for model_name, history in results.items():
            history_dict = history.history

            summary["models"][model_name] = {
                "final_loss": float(history_dict["loss"][-1]),
                "final_val_loss": float(history_dict["val_loss"][-1]),
                "best_val_loss": float(min(history_dict["val_loss"])),
                "final_accuracy": float(history_dict.get("accuracy", [0])[-1]),
                "final_val_accuracy": float(history_dict.get("val_accuracy", [0])[-1]),
                "best_val_accuracy": float(max(history_dict.get("val_accuracy", [0]))),
            }

        filepath = self.config.get_output_path() / "training_summary.json"
        save_json(summary, filepath)

        logger.info(f"Saved training summary to {filepath}")

    def get_best_model(self, metric: str = "val_accuracy", mode: str = "max") -> Tuple[str, float]:
        """
        Get the best performing model.

        Args:
            metric: Metric to compare
            mode: 'max' for accuracy, 'min' for loss

        Returns:
            Tuple of (model_name, best_value)
        """
        if not self.training_history:
            raise ValueError("No training history available")

        best_model = None
        best_value = float("-inf") if mode == "max" else float("inf")

        for model_name, info in self.training_history.items():
            history = info["history"]

            if metric not in history:
                logger.warning(f"Metric '{metric}' not found for {model_name}")
                continue

            if mode == "max":
                value = max(history[metric])
                if value > best_value:
                    best_value = value
                    best_model = model_name
            else:
                value = min(history[metric])
                if value < best_value:
                    best_value = value
                    best_model = model_name

        logger.info(f"Best model: {best_model} with {metric}={best_value:.4f}")
        return best_model, best_value
