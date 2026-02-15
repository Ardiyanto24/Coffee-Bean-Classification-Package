"""Metrics calculation utilities."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score
)
from typing import Dict, Any, Optional, List
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Calculate various classification metrics.
    
    Provides comprehensive metrics calculation including:
    - Accuracy (overall, per-class, balanced)
    - Precision, Recall, F1-score
    - Confusion matrix
    - ROC-AUC
    - Top-K accuracy
    
    Example:
        >>> calculator = MetricsCalculator(class_names=['A', 'B', 'C'])
        >>> metrics = calculator.calculate_all(y_true, y_pred)
        >>> print(metrics['accuracy'])
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        logger.info(f"MetricsCalculator initialized for {self.num_classes} classes")
    
    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True labels (one-hot or class indices)
            y_pred: Predicted labels (one-hot or class indices)
            y_pred_proba: Prediction probabilities (for ROC-AUC)
            
        Returns:
            Dictionary with all metrics
        """
        # Convert to class indices if one-hot
        if len(y_true.shape) > 1:
            y_true_idx = np.argmax(y_true, axis=1)
        else:
            y_true_idx = y_true
        
        if len(y_pred.shape) > 1:
            y_pred_idx = np.argmax(y_pred, axis=1)
        else:
            y_pred_idx = y_pred
        
        metrics = {
            'accuracy': self.calculate_accuracy(y_true_idx, y_pred_idx),
            'balanced_accuracy': self.calculate_balanced_accuracy(y_true_idx, y_pred_idx),
            'precision_macro': self.calculate_precision(y_true_idx, y_pred_idx, 'macro'),
            'precision_weighted': self.calculate_precision(y_true_idx, y_pred_idx, 'weighted'),
            'recall_macro': self.calculate_recall(y_true_idx, y_pred_idx, 'macro'),
            'recall_weighted': self.calculate_recall(y_true_idx, y_pred_idx, 'weighted'),
            'f1_macro': self.calculate_f1(y_true_idx, y_pred_idx, 'macro'),
            'f1_weighted': self.calculate_f1(y_true_idx, y_pred_idx, 'weighted'),
        }
        
        # Per-class metrics
        per_class = self.calculate_per_class_metrics(y_true_idx, y_pred_idx)
        metrics['per_class'] = per_class
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true_idx, y_pred_idx)
        
        # ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                # Convert y_true to one-hot if needed
                if len(y_true.shape) == 1:
                    y_true_onehot = tf.keras.utils.to_categorical(y_true_idx, self.num_classes)
                else:
                    y_true_onehot = y_true
                
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true_onehot,
                    y_pred_proba,
                    average='macro',
                    multi_class='ovr'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true_onehot,
                    y_pred_proba,
                    average='weighted',
                    multi_class='ovr'
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Top-K accuracy if probabilities provided
        if y_pred_proba is not None:
            metrics['top_2_accuracy'] = self.calculate_top_k_accuracy(
                y_true_idx, y_pred_proba, k=2
            )
            if self.num_classes >= 3:
                metrics['top_3_accuracy'] = self.calculate_top_k_accuracy(
                    y_true_idx, y_pred_proba, k=3
                )
        
        logger.info(f"Calculated {len(metrics)} metrics")
        return metrics
    
    @staticmethod
    def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate overall accuracy."""
        return float(accuracy_score(y_true, y_pred))
    
    @staticmethod
    def calculate_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy (useful for imbalanced datasets)."""
        return float(balanced_accuracy_score(y_true, y_pred))
    
    @staticmethod
    def calculate_precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """Calculate precision score."""
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """Calculate recall score."""
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'macro'
    ) -> float:
        """Calculate F1 score."""
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    def calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate metrics for each class.
        
        Returns:
            DataFrame with per-class metrics
        """
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Count support (number of samples per class)
        support = np.bincount(y_true, minlength=self.num_classes)
        
        df = pd.DataFrame({
            'class': self.class_names,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        })
        
        return df
    
    @staticmethod
    def calculate_top_k_accuracy(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        k: int = 2
    ) -> float:
        """
        Calculate top-K accuracy.
        
        Args:
            y_true: True class indices
            y_pred_proba: Prediction probabilities
            k: Number of top predictions to consider
            
        Returns:
            Top-K accuracy
        """
        # Get top-k predictions
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        
        # Check if true label is in top-k
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get sklearn classification report as dictionary.
        
        Returns:
            Classification report dictionary
        """
        # Convert to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    def calculate_confusion_matrix_normalized(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate normalized confusion matrix (percentages).
        
        Returns:
            Normalized confusion matrix
        """
        # Convert to class indices if needed
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm_normalized
