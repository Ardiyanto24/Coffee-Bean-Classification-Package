"""Classification model evaluator."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from .metrics import MetricsCalculator
from ..utils import get_logger, ensure_dir

logger = get_logger(__name__)


class ClassificationEvaluator:
    """
    Evaluate classification models.
    
    Provides:
    - Comprehensive metrics calculation
    - Confusion matrix plotting
    - ROC curves
    - Prediction analysis
    - Report generation
    
    Example:
        >>> evaluator = ClassificationEvaluator(model, class_names)
        >>> metrics = evaluator.evaluate(test_dataset)
        >>> evaluator.plot_confusion_matrix(save_path='cm.png')
        >>> report = evaluator.generate_report()
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        class_names: List[str],
        output_dir: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            class_names: List of class names
            output_dir: Directory for saving outputs
        """
        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        if output_dir:
            self.output_dir = ensure_dir(output_dir)
        else:
            self.output_dir = Path('evaluation_outputs')
            ensure_dir(self.output_dir)
        
        self.metrics_calculator = MetricsCalculator(class_names)
        
        # Storage for evaluation results
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = None
        
        logger.info(f"ClassificationEvaluator initialized for {self.num_classes} classes")
    
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: TensorFlow dataset
            verbose: Verbosity level
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Evaluating model...")
        
        # Get predictions
        y_true_list = []
        y_pred_proba_list = []
        
        for images, labels in dataset:
            predictions = self.model.predict(images, verbose=0)
            y_true_list.append(labels.numpy())
            y_pred_proba_list.append(predictions)
        
        # Concatenate all batches
        self.y_true = np.concatenate(y_true_list, axis=0)
        self.y_pred_proba = np.concatenate(y_pred_proba_list, axis=0)
        self.y_pred = (self.y_pred_proba == self.y_pred_proba.max(axis=1, keepdims=True)).astype(int)
        
        # Calculate metrics
        self.metrics = self.metrics_calculator.calculate_all(
            self.y_true,
            self.y_pred,
            self.y_pred_proba
        )
        
        if verbose:
            self._print_metrics_summary()
        
        logger.info("✓ Evaluation completed")
        return self.metrics
    
    def _print_metrics_summary(self) -> None:
        """Print summary of main metrics."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy:           {self.metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {self.metrics['balanced_accuracy']:.4f}")
        print(f"Precision (macro):  {self.metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {self.metrics['recall_macro']:.4f}")
        print(f"F1-Score (macro):   {self.metrics['f1_macro']:.4f}")
        
        if 'roc_auc_macro' in self.metrics:
            print(f"ROC-AUC (macro):    {self.metrics['roc_auc_macro']:.4f}")
        
        if 'top_2_accuracy' in self.metrics:
            print(f"Top-2 Accuracy:     {self.metrics['top_2_accuracy']:.4f}")
        
        print("=" * 60 + "\n")
    
    def plot_confusion_matrix(
        self,
        normalized: bool = True,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            normalized: If True, show percentages
            save_path: Path to save figure
            figsize: Figure size
        """
        if self.y_true is None:
            raise ValueError("Run evaluate() first")
        
        plt.figure(figsize=figsize)
        
        if normalized:
            cm = self.metrics_calculator.calculate_confusion_matrix_normalized(
                self.y_true, self.y_pred
            )
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            cm = self.metrics['confusion_matrix']
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage' if normalized else 'Count'}
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ) -> None:
        """
        Plot per-class precision, recall, and F1-score.
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        if self.metrics is None:
            raise ValueError("Run evaluate() first")
        
        per_class = self.metrics['per_class']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot for metrics
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax1.bar(x - width, per_class['precision'], width, label='Precision', alpha=0.8)
        ax1.bar(x, per_class['recall'], width, label='Recall', alpha=0.8)
        ax1.bar(x + width, per_class['f1_score'], width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Class', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Support (number of samples per class)
        ax2.bar(self.class_names, per_class['support'], color='steelblue', alpha=0.7)
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved per-class metrics to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(
        self,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ) -> None:
        """
        Plot ROC curves for each class.
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        if self.y_pred_proba is None:
            raise ValueError("Run evaluate() first")
        
        from sklearn.metrics import roc_curve, auc
        
        # Convert to one-hot if needed
        if len(self.y_true.shape) == 1:
            y_true_onehot = tf.keras.utils.to_categorical(
                np.argmax(self.y_true, axis=1) if len(self.y_true.shape) > 1 else self.y_true,
                self.num_classes
            )
        else:
            y_true_onehot = self.y_true
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], self.y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curves to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report as string
        """
        if self.metrics is None:
            raise ValueError("Run evaluate() first")
        
        # Build report
        lines = []
        lines.append("=" * 70)
        lines.append("MODEL EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Overall metrics
        lines.append("OVERALL METRICS:")
        lines.append("-" * 70)
        lines.append(f"Accuracy:              {self.metrics['accuracy']:.4f}")
        lines.append(f"Balanced Accuracy:     {self.metrics['balanced_accuracy']:.4f}")
        lines.append(f"Precision (macro):     {self.metrics['precision_macro']:.4f}")
        lines.append(f"Precision (weighted):  {self.metrics['precision_weighted']:.4f}")
        lines.append(f"Recall (macro):        {self.metrics['recall_macro']:.4f}")
        lines.append(f"Recall (weighted):     {self.metrics['recall_weighted']:.4f}")
        lines.append(f"F1-Score (macro):      {self.metrics['f1_macro']:.4f}")
        lines.append(f"F1-Score (weighted):   {self.metrics['f1_weighted']:.4f}")
        
        if 'roc_auc_macro' in self.metrics:
            lines.append(f"ROC-AUC (macro):       {self.metrics['roc_auc_macro']:.4f}")
            lines.append(f"ROC-AUC (weighted):    {self.metrics['roc_auc_weighted']:.4f}")
        
        if 'top_2_accuracy' in self.metrics:
            lines.append(f"Top-2 Accuracy:        {self.metrics['top_2_accuracy']:.4f}")
        if 'top_3_accuracy' in self.metrics:
            lines.append(f"Top-3 Accuracy:        {self.metrics['top_3_accuracy']:.4f}")
        
        lines.append("")
        
        # Per-class metrics
        lines.append("PER-CLASS METRICS:")
        lines.append("-" * 70)
        per_class = self.metrics['per_class']
        
        lines.append(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        lines.append("-" * 70)
        
        for _, row in per_class.iterrows():
            lines.append(
                f"{row['class']:<15} "
                f"{row['precision']:<12.4f} "
                f"{row['recall']:<12.4f} "
                f"{row['f1_score']:<12.4f} "
                f"{int(row['support']):<10}"
            )
        
        lines.append("")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        # Print report
        print(report)
        
        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved report to {save_path}")
        
        return report
    
    def get_misclassified_samples(
        self,
        dataset: tf.data.Dataset,
        max_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get information about misclassified samples.
        
        Args:
            dataset: Original dataset
            max_samples: Maximum number of samples to return
            
        Returns:
            List of misclassified sample information
        """
        if self.y_true is None:
            raise ValueError("Run evaluate() first")
        
        # Get class indices
        y_true_idx = np.argmax(self.y_true, axis=1) if len(self.y_true.shape) > 1 else self.y_true
        y_pred_idx = np.argmax(self.y_pred, axis=1) if len(self.y_pred.shape) > 1 else self.y_pred
        
        # Find misclassified indices
        misclassified_idx = np.where(y_true_idx != y_pred_idx)[0]
        
        logger.info(f"Found {len(misclassified_idx)} misclassified samples")
        
        # Limit number of samples
        misclassified_idx = misclassified_idx[:max_samples]
        
        # Get details
        misclassified = []
        for idx in misclassified_idx:
            misclassified.append({
                'index': int(idx),
                'true_class': self.class_names[y_true_idx[idx]],
                'predicted_class': self.class_names[y_pred_idx[idx]],
                'confidence': float(self.y_pred_proba[idx, y_pred_idx[idx]]),
                'probabilities': {
                    self.class_names[i]: float(self.y_pred_proba[idx, i])
                    for i in range(self.num_classes)
                }
            })
        
        return misclassified
