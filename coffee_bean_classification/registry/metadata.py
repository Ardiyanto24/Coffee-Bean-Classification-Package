"""Model metadata management."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import json


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    model_name: str
    version: str
    architecture: str
    created_at: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    input_shape: tuple
    num_classes: int
    tags: List[str]
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelMetadata":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
