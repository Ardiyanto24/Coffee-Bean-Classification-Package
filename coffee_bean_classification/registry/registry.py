"""Model registry for version control and management."""

import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional, List
import json
from datetime import datetime

from .metadata import ModelMetadata
from ..utils import get_logger, ensure_dir, save_json, load_json

logger = get_logger(__name__)

class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self, registry_path: str = "model_registry"):
        """Initialize model registry."""
        self.registry_path = Path(registry_path)
        ensure_dir(self.registry_path)
        self.models_dir = ensure_dir(self.registry_path / "models")
        self.metadata_file = self.registry_path / "registry.json"
        self._load_registry()
        logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def _load_registry(self):
        """Load registry from disk."""
        if self.metadata_file.exists():
            self._registry = load_json(self.metadata_file)
        else:
            self._registry = {}
    
    def _save_registry(self):
        """Save registry to disk."""
        save_json(self._registry, self.metadata_file)
    
    def register_model(
        self,
        model: tf.keras.Model,
        metadata: ModelMetadata
    ) -> str:
        """Register a model with metadata."""
        model_id = f"{metadata.model_name}_{metadata.version}"
        model_path = self.models_dir / f"{model_id}.h5"
        
        # Save model
        model.save(model_path)
        
        # Save metadata
        self._registry[model_id] = metadata.to_dict()
        self._save_registry()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def load_model(self, model_name: str, version: str = "latest") -> tf.keras.Model:
        """Load a model from registry."""
        if version == "latest":
            version = self._get_latest_version(model_name)
        
        model_id = f"{model_name}_{version}"
        if model_id not in self._registry:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = self.models_dir / f"{model_id}.h5"
        model = tf.keras.models.load_model(model_path)
        
        logger.info(f"Loaded model: {model_id}")
        return model
    
    def get_best_model(self, metric: str = "accuracy") -> str:
        """Get best model by metric."""
        best_id = None
        best_value = -1
        
        for model_id, meta in self._registry.items():
            if metric in meta['metrics']:
                value = meta['metrics'][metric]
                if value > best_value:
                    best_value = value
                    best_id = model_id
        
        logger.info(f"Best model: {best_id} ({metric}={best_value})")
        return best_id
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._registry.keys())
    
    def _get_latest_version(self, model_name: str) -> str:
        """Get latest version of a model."""
        versions = [
            k.split('_')[-1] for k in self._registry.keys()
            if k.startswith(model_name)
        ]
        if not versions:
            raise ValueError(f"No versions found for {model_name}")
        return sorted(versions)[-1]
