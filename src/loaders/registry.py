from pathlib import Path
from typing import Set, Dict
from uuid import uuid4
from collections import defaultdict
from models.appearance_feature_extractor import AppearanceFeatureExtractor

WEIGHT_ROOT = Path(__file__).resolve().parent / "weights"


class ModelRegistry:
    _registry = {}
    _alias_map = {}

    @classmethod
    def register(cls, names: Set[str], data: Dict):
        r_id = uuid4()
        cls._registry[r_id] = data
        for name in names:
            cls._alias_map[name] = r_id

    @classmethod
    def get_path(cls, name: str) -> Path:
        if name not in cls._alias_map:
            raise KeyError(f"Model '{name}' not found in registry.")

        model = cls._registry.get(cls._alias_map[name], None)
        # Second check just to be sure
        if not model:
            raise ValueError("Expected model object, got None")
        return model

    @classmethod
    def list(cls):
        reference = defaultdict(list)
        for k, v in cls._alias_map.items():
            reference[v].append(k)

        return {str(v): str(cls._registry[k]) for k, v in reference.items()}


ModelRegistry.register(
    {"appearance_feature_extractor", "E_H", "E_C"},
    {
        "model_builder": AppearanceFeatureExtractor,
        "weight_source": {
            "type": "huggingface",
            "repo": "KlingTeam/LivePortrait",
            "repo_type": "space",
            "path": "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
            "save_to": "appearance_feature_extractor.pth"
        },
        "loader": "pytorch_loader",
        "key_mapper": "default",  # Not implemented yet
        "precision": "fp32"       # Not implemented yet
    }
)

if __name__ == "__main__":
    print(ModelRegistry.list())
