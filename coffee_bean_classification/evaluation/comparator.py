"""Model comparison utilities."""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import get_logger, save_json

logger = get_logger(__name__)


class ModelComparator:
    """
    Compare multiple trained models.

    Provides:
    - Performance comparison
    - Ranking models
    - Comparison charts
    - Leaderboard generation

    Example:
        >>> comparator = ModelComparator()
        >>> comparator.add_model('resnet50', metrics1)
        >>> comparator.add_model('efficientnet', metrics2)
        >>> df = comparator.get_comparison_df()
        >>> comparator.plot_comparison()
    """

    def __init__(self):
        """Initialize model comparator."""
        self.models = {}
        logger.info("ModelComparator initialized")

    def add_model(self, model_name: str, metrics: Dict[str, Any]) -> None:
        """
        Add a model's metrics to comparison.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
        """
        self.models[model_name] = metrics
        logger.info(f"Added {model_name} to comparison")

    def get_comparison_df(self, metrics_to_compare: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get comparison as DataFrame.

        Args:
            metrics_to_compare: List of metric names to include

        Returns:
            DataFrame with comparison
        """
        if not self.models:
            raise ValueError("No models added for comparison")

        if metrics_to_compare is None:
            metrics_to_compare = [
                "accuracy",
                "balanced_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "roc_auc_macro",
            ]

        # Build comparison data
        data = []
        for model_name, metrics in self.models.items():
            row = {"model": model_name}
            for metric in metrics_to_compare:
                if metric in metrics:
                    row[metric] = metrics[metric]
                else:
                    row[metric] = None
            data.append(row)

        df = pd.DataFrame(data)
        df = df.set_index("model")

        logger.info(f"Created comparison with {len(df)} models")
        return df

    def rank_models(self, metric: str = "accuracy", ascending: bool = False) -> pd.DataFrame:
        """
        Rank models by a specific metric.

        Args:
            metric: Metric to rank by
            ascending: If True, rank ascending (for loss)

        Returns:
            Ranked DataFrame
        """
        df = self.get_comparison_df()

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found")

        ranked = df.sort_values(by=metric, ascending=ascending)
        ranked["rank"] = range(1, len(ranked) + 1)

        logger.info(f"Ranked models by {metric}")
        return ranked

    def get_best_model(self, metric: str = "accuracy") -> tuple:
        """
        Get the best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, metric_value)
        """
        ranked = self.rank_models(metric, ascending=False)
        best_model = ranked.index[0]
        best_value = ranked.iloc[0][metric]

        logger.info(f"Best model: {best_model} ({metric}={best_value:.4f})")
        return best_model, best_value

    def plot_comparison(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6),
    ) -> None:
        """
        Plot comparison bar chart.

        Args:
            metrics: Metrics to plot
            save_path: Path to save figure
            figsize: Figure size
        """
        if metrics is None:
            metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        df = self.get_comparison_df(metrics)

        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            df[metric].plot(kind="bar", ax=ax, color="steelblue", alpha=0.7)
            ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
            ax.set_ylabel("Score")
            ax.set_ylim([0, 1])
            ax.grid(axis="y", alpha=0.3)
            ax.set_xticklabels(df.index, rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")

        plt.show()

    def plot_radar_chart(self, save_path: Optional[str] = None, figsize: tuple = (10, 10)) -> None:
        """
        Plot radar chart for model comparison.

        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        df = self.get_comparison_df(metrics)

        # Radar chart setup
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        for model_name in df.index:
            values = df.loc[model_name, metrics].tolist()
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.title("Model Comparison - Radar Chart", size=16, fontweight="bold", pad=20)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved radar chart to {save_path}")

        plt.show()

    def export_leaderboard(self, save_path: str, metric: str = "f1_macro") -> None:
        """
        Export leaderboard to file.

        Args:
            save_path: Path to save leaderboard
            metric: Metric to rank by
        """
        ranked = self.rank_models(metric)

        if save_path.endswith(".csv"):
            ranked.to_csv(save_path)
        elif save_path.endswith(".json"):
            save_json(ranked.to_dict("index"), save_path)
        else:
            # Default to CSV
            ranked.to_csv(save_path)

        logger.info(f"Exported leaderboard to {save_path}")
