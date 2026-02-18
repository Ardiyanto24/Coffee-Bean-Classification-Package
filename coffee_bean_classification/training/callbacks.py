"""Callback manager for training."""

from datetime import datetime
from typing import List, Optional

import tensorflow as tf

from ..utils import ensure_dir, get_logger

logger = get_logger(__name__)


class CallbackManager:
    """
    Manages callbacks for model training.

    Creates and configures callbacks based on configuration settings.

    Supported callbacks:
    - ModelCheckpoint: Save best models
    - EarlyStopping: Stop when no improvement
    - ReduceLROnPlateau: Reduce learning rate
    - TensorBoard: Logging for visualization
    - CSVLogger: Log metrics to CSV
    - LearningRateScheduler: Custom LR schedules
    - TerminateOnNaN: Stop if loss is NaN

    Example:
        >>> manager = CallbackManager(config)
        >>> callbacks = manager.get_callbacks(model_name='resnet50')
        >>> model.fit(train_ds, callbacks=callbacks)
    """

    def __init__(self, config):
        """
        Initialize callback manager.

        Args:
            config: TrainingConfig instance
        """
        self.config = config
        self.callbacks_config = config.callbacks

        # Create output directories
        self.checkpoint_dir = ensure_dir(config.get_output_path("checkpoints"))
        self.logs_dir = ensure_dir(config.get_output_path("logs"))
        self.tensorboard_dir = ensure_dir(config.get_output_path("tensorboard"))

        logger.info("CallbackManager initialized")

    def get_callbacks(
        self,
        model_name: str = "model",
        custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Get list of configured callbacks.

        Args:
            model_name: Name of the model (for file naming)
            custom_callbacks: Additional custom callbacks

        Returns:
            List of Keras callbacks
        """
        callbacks = []

        # ModelCheckpoint
        if self.callbacks_config.get("checkpoint", True):
            callbacks.append(self._create_checkpoint_callback(model_name))

        # EarlyStopping
        if self.callbacks_config.get("early_stopping", True):
            callbacks.append(self._create_early_stopping_callback())

        # ReduceLROnPlateau
        if self.callbacks_config.get("reduce_lr", True):
            callbacks.append(self._create_reduce_lr_callback())

        # TensorBoard
        if self.callbacks_config.get("tensorboard", False):
            callbacks.append(self._create_tensorboard_callback(model_name))

        # CSVLogger
        if self.callbacks_config.get("csv_logger", True):
            callbacks.append(self._create_csv_logger_callback(model_name))

        # TerminateOnNaN
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())

        # Add custom callbacks
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
            logger.info(f"Added {len(custom_callbacks)} custom callbacks")

        logger.info(f"Created {len(callbacks)} callbacks for {model_name}")
        return callbacks

    def _create_checkpoint_callback(self, model_name: str) -> tf.keras.callbacks.ModelCheckpoint:
        """Create ModelCheckpoint callback."""
        filepath = self.checkpoint_dir / f"{model_name}_best.h5"

        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(filepath),
            monitor=self.config.checkpoint_monitor,
            mode=self.config.checkpoint_mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )

        logger.debug(f"ModelCheckpoint: {filepath}")
        return callback

    def _create_early_stopping_callback(self) -> tf.keras.callbacks.EarlyStopping:
        """Create EarlyStopping callback."""
        callback = tf.keras.callbacks.EarlyStopping(
            monitor=self.config.checkpoint_monitor,
            patience=self.config.early_stopping_patience,
            mode=self.config.checkpoint_mode,
            restore_best_weights=True,
            verbose=1,
        )

        logger.debug(f"EarlyStopping: patience={self.config.early_stopping_patience}")
        return callback

    def _create_reduce_lr_callback(self) -> tf.keras.callbacks.ReduceLROnPlateau:
        """Create ReduceLROnPlateau callback."""
        callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.config.checkpoint_monitor,
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            mode=self.config.checkpoint_mode,
            min_lr=1e-7,
            verbose=1,
        )

        logger.debug(
            f"ReduceLROnPlateau: factor={self.config.reduce_lr_factor}, "
            f"patience={self.config.reduce_lr_patience}"
        )
        return callback

    def _create_tensorboard_callback(self, model_name: str) -> tf.keras.callbacks.TensorBoard:
        """Create TensorBoard callback."""
        log_dir = self.tensorboard_dir / model_name / datetime.now().strftime("%Y%m%d-%H%M%S")
        ensure_dir(log_dir)

        callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
            profile_batch=0,
        )

        logger.debug(f"TensorBoard: {log_dir}")
        return callback

    def _create_csv_logger_callback(self, model_name: str) -> tf.keras.callbacks.CSVLogger:
        """Create CSVLogger callback."""
        filename = self.logs_dir / f"{model_name}_training.csv"

        callback = tf.keras.callbacks.CSVLogger(filename=str(filename), separator=",", append=False)

        logger.debug(f"CSVLogger: {filename}")
        return callback

    def create_learning_rate_scheduler(
        self, schedule_type: str = "exponential", initial_lr: Optional[float] = None, **kwargs
    ) -> tf.keras.callbacks.LearningRateScheduler:
        """
        Create a learning rate scheduler callback.

        Args:
            schedule_type: Type of schedule ('exponential', 'step', 'cosine')
            initial_lr: Initial learning rate
            **kwargs: Additional arguments for the schedule

        Returns:
            LearningRateScheduler callback
        """
        if initial_lr is None:
            initial_lr = self.config.learning_rate

        if schedule_type == "exponential":
            decay_rate = kwargs.get("decay_rate", 0.96)
            decay_steps = kwargs.get("decay_steps", 1000)

            def schedule(epoch, lr):
                return initial_lr * (decay_rate ** (epoch / decay_steps))

        elif schedule_type == "step":
            drop_rate = kwargs.get("drop_rate", 0.5)
            drop_every = kwargs.get("drop_every", 10)

            def schedule(epoch, lr):
                return initial_lr * (drop_rate ** (epoch // drop_every))

        elif schedule_type == "cosine":
            import numpy as np

            total_epochs = kwargs.get("total_epochs", self.config.epochs)

            def schedule(epoch, lr):
                return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        callback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)
        logger.info(f"Created {schedule_type} learning rate scheduler")

        return callback

    def create_custom_callback(
        self,
        on_epoch_end: Optional[callable] = None,
        on_batch_end: Optional[callable] = None,
        on_train_begin: Optional[callable] = None,
        on_train_end: Optional[callable] = None,
    ) -> tf.keras.callbacks.LambdaCallback:
        """
        Create a custom callback with lambda functions.

        Args:
            on_epoch_end: Function called at end of each epoch
            on_batch_end: Function called at end of each batch
            on_train_begin: Function called at start of training
            on_train_end: Function called at end of training

        Returns:
            LambdaCallback instance
        """
        callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=on_epoch_end,
            on_batch_end=on_batch_end,
            on_train_begin=on_train_begin,
            on_train_end=on_train_end,
        )

        logger.info("Created custom lambda callback")
        return callback


class ProgressCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for progress tracking and logging.

    Provides:
    - Epoch progress tracking
    - ETA calculation
    - Metric logging
    - Custom notifications
    """

    def __init__(self, total_epochs: int, model_name: str = "model"):
        """
        Initialize progress callback.

        Args:
            total_epochs: Total number of epochs
            model_name: Name of the model
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.epoch_start_time = None

        logger.info(f"ProgressCallback initialized for {model_name}")

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch."""
        import time

        self.epoch_start_time = time.time()
        logger.info(f"[{self.model_name}] Epoch {epoch + 1}/{self.total_epochs} started")

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        import time

        if logs is None:
            logs = {}

        epoch_time = time.time() - self.epoch_start_time

        # Calculate ETA
        epochs_remaining = self.total_epochs - (epoch + 1)
        eta_seconds = epoch_time * epochs_remaining
        eta_minutes = eta_seconds / 60

        # Format metrics
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])

        logger.info(
            f"[{self.model_name}] Epoch {epoch + 1}/{self.total_epochs} - "
            f"Time: {epoch_time:.1f}s - ETA: {eta_minutes:.1f}min - {metrics_str}"
        )

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        logger.info(f"[{self.model_name}] Training completed!")
