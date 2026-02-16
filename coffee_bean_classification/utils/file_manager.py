"""File and directory management utilities."""

import json
import yaml
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path

    Returns:
        Path object

    Example:
        >>> output_dir = ensure_dir('models/checkpoints')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def ensure_parent_dir(file_path: Union[str, Path]) -> Path:
    """
    Create parent directory of a file if it doesn't exist.

    Args:
        file_path: File path

    Returns:
        Parent directory Path object
    """
    file_path = Path(file_path)
    if file_path.parent != Path("."):
        ensure_dir(file_path.parent)
    return file_path.parent


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> Path:
    """
    Save dictionary as JSON file.

    Args:
        data: Dictionary to save
        path: Output file path
        indent: JSON indentation

    Returns:
        Path to saved file

    Example:
        >>> config = {'epochs': 50, 'batch_size': 32}
        >>> save_json(config, 'config.json')
    """
    path = Path(path)
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

    logger.debug(f"Saved JSON to: {path}")
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.

    Args:
        path: JSON file path

    Returns:
        Dictionary from JSON

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        >>> config = load_json('config.json')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from: {path}")
    return data


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """
    Save dictionary as YAML file.

    Args:
        data: Dictionary to save
        path: Output file path

    Returns:
        Path to saved file

    Example:
        >>> config = {'epochs': 50, 'batch_size': 32}
        >>> save_yaml(config, 'config.yaml')
    """
    path = Path(path)
    ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    logger.debug(f"Saved YAML to: {path}")
    return path


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML file as dictionary.

    Args:
        path: YAML file path

    Returns:
        Dictionary from YAML

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        >>> config = load_yaml('config.yaml')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.debug(f"Loaded YAML from: {path}")
    return data


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Copy file from source to destination.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        Path to destination file
    """
    src = Path(src)
    dst = Path(dst)
    ensure_parent_dir(dst)

    shutil.copy2(src, dst)
    logger.debug(f"Copied {src} -> {dst}")
    return dst


def copy_directory(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Copy directory recursively.

    Args:
        src: Source directory path
        dst: Destination directory path

    Returns:
        Path to destination directory
    """
    src = Path(src)
    dst = Path(dst)

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    logger.debug(f"Copied directory {src} -> {dst}")
    return dst


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        path: File path

    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")

    Example:
        >>> format_size(1536000)
        '1.46 MB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def create_versioned_dir(base_path: Union[str, Path], prefix: str = "v") -> Path:
    """
    Create a versioned directory.

    Args:
        base_path: Base directory path
        prefix: Version prefix

    Returns:
        Path to versioned directory

    Example:
        >>> # Creates 'experiments/v1', 'experiments/v2', etc.
        >>> exp_dir = create_versioned_dir('experiments')
    """
    base_path = Path(base_path)
    ensure_dir(base_path)

    # Find existing versions
    existing = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith(prefix)]

    if not existing:
        version = 1
    else:
        versions = []
        for d in existing:
            try:
                v = int(d.name[len(prefix) :])
                versions.append(v)
            except ValueError:
                continue
        version = max(versions, default=0) + 1

    versioned_path = base_path / f"{prefix}{version}"
    ensure_dir(versioned_path)

    logger.info(f"Created versioned directory: {versioned_path}")
    return versioned_path


def create_timestamped_dir(base_path: Union[str, Path], prefix: str = "") -> Path:
    """
    Create a timestamped directory.

    Args:
        base_path: Base directory path
        prefix: Optional prefix before timestamp

    Returns:
        Path to timestamped directory

    Example:
        >>> # Creates 'runs/experiment_20240214_153045'
        >>> run_dir = create_timestamped_dir('runs', 'experiment')
    """
    base_path = Path(base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if prefix:
        dir_name = f"{prefix}_{timestamp}"
    else:
        dir_name = timestamp

    timestamped_path = base_path / dir_name
    ensure_dir(timestamped_path)

    logger.info(f"Created timestamped directory: {timestamped_path}")
    return timestamped_path


class FileManager:
    """
    Centralized file management for experiments.

    Manages directory structure for training runs, including:
    - Models/checkpoints
    - Logs
    - Metrics
    - Plots
    - Configs
    """

    def __init__(self, base_dir: Union[str, Path], experiment_name: str = "experiment"):
        """
        Initialize file manager.

        Args:
            base_dir: Base directory for all outputs
            experiment_name: Name of the experiment
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name

        # Create timestamped run directory
        self.run_dir = create_timestamped_dir(self.base_dir, experiment_name)

        # Create subdirectories
        self.models_dir = ensure_dir(self.run_dir / "models")
        self.logs_dir = ensure_dir(self.run_dir / "logs")
        self.metrics_dir = ensure_dir(self.run_dir / "metrics")
        self.plots_dir = ensure_dir(self.run_dir / "plots")
        self.configs_dir = ensure_dir(self.run_dir / "configs")

        logger.info(f"FileManager initialized at: {self.run_dir}")

    def get_model_path(self, model_name: str, suffix: str = "best") -> Path:
        """Get path for saving model."""
        return self.models_dir / f"{model_name}_{suffix}.h5"

    def get_log_path(self, log_name: str = "training") -> Path:
        """Get path for log file."""
        return self.logs_dir / f"{log_name}.log"

    def get_metrics_path(self, metrics_name: str = "metrics") -> Path:
        """Get path for metrics file."""
        return self.metrics_dir / f"{metrics_name}.json"

    def get_plot_path(self, plot_name: str) -> Path:
        """Get path for plot file."""
        return self.plots_dir / f"{plot_name}.png"

    def get_config_path(self, config_name: str = "config") -> Path:
        """Get path for config file."""
        return self.configs_dir / f"{config_name}.yaml"

    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Save experiment metadata."""
        metadata_path = self.run_dir / "metadata.json"
        return save_json(metadata, metadata_path)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of managed files."""
        return {
            "run_dir": str(self.run_dir),
            "experiment_name": self.experiment_name,
            "subdirectories": {
                "models": str(self.models_dir),
                "logs": str(self.logs_dir),
                "metrics": str(self.metrics_dir),
                "plots": str(self.plots_dir),
                "configs": str(self.configs_dir),
            },
            "created_at": datetime.now().isoformat(),
        }
