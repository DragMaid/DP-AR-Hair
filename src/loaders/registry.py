from pathlib import Path
from typing import Set, Dict
from uuid import uuid4
from collections import defaultdict
import importlib

from configs.model_config import model_config

ROOT_DIR = Path(__file__).resolve().parents[2]
WEIGHT_ROOT = ROOT_DIR / "weights"


def _resolve_builder(path: str):
    """
    Lazily resolve a class from a string import path.
    """
    module_path, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


class ModelRegistry:
    _registry: Dict = {}
    _alias_map: Dict = {}

    @classmethod
    def register(cls, names: Set[str], data: Dict):
        r_id = uuid4()
        cls._registry[r_id] = data
        for name in names:
            cls._alias_map[name] = r_id

    @classmethod
    def get_registry(cls, name: str) -> dict:
        if name not in cls._alias_map:
            raise KeyError(f"Model '{name}' not found in registry.")

        registry = cls._registry.get(cls._alias_map[name])
        if registry is None:
            raise ValueError(f"Expected model registry, got {registry}")

        # Resolve model builder lazily (only once)
        builder = registry.get("model_builder")
        if isinstance(builder, str):
            resolved = _resolve_builder(builder)
            registry["model_builder"] = resolved

        return registry

    @classmethod
    def list(cls):
        reference = defaultdict(list)
        for alias, r_id in cls._alias_map.items():
            reference[r_id].append(alias)

        return {
            str(r_id): str(cls._registry[r_id])
            for r_id in reference
        }


# -----------------------
# Registrations
# -----------------------

ModelRegistry.register(
    {"appearance_feature_extractor", "E_H"},
    {
        "model_builder": "live_portrait.models.appearance_feature_extractor.AppearanceFeatureExtractor",
        "params": model_config.appearance_feature_extractor_params,
        "weight": {
            "type": "hf_file",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",
        "precision": "fp32",
    },
)

ModelRegistry.register(
    {"motion_extractor", "E_M"},
    {
        "model_builder": "live_portrait.models.motion_extractor.MotionExtractor",
        "params": model_config.motion_extractor_params,
        "weight": {
            "type": "hf_file",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",
        "precision": "fp32",
    },
)

ModelRegistry.register(
    {"warping_module", "W"},
    {
        "model_builder": "live_portrait.models.warping_network.WarpingNetwork",
        "params": model_config.warping_module_params,
        "weight": {
            "type": "hf_file",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/warping_module.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",
        "precision": "fp32",
    },
)

ModelRegistry.register(
    {"spade_generator", "context_decoder", "D_C"},
    {
        "model_builder": "live_portrait.models.context_decoder.ContextDecoder",
        "params": model_config.context_decoder_params,
        "weight": {
            "type": "hf_file",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/spade_generator.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",
        "precision": "fp32",
    },
)

ModelRegistry.register(
    {"Hair Mask", "M_C"},
    {
        "model_builder": "face_parsing.models.bisenet.BiSeNet",
        "params": model_config.face_parsing_params,
        "weight": {
            "type": "direct_link",
            "options": {
                "link": "https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet18.pt",
                "filename": "resnet18.pt",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",
        "precision": "fp32",
    },
)

ModelRegistry.register(
    {"synthesis_decoder", "D_S"},
    {
        "model_builder": "models.synthesis_decoder.SynthesisDecoder",
        "params": model_config.synthesis_decoder_params,
        "weight": {
            "type": "hf_file",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/spade_generator.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",
        "precision": "fp32",
    },
)

ModelRegistry.register(
    {"context_encoder", "E_C"},
    {
        "model_builder": "models.context_encoder.ContextEncoder",
        "params": model_config.context_encoder_params,
        "loader": "pytorch",
        "key_mapper": "default",
        "precision": "fp32",
    },
)

ModelRegistry.register(
    {"gan_hair", "IIHT1"},
    {
        "model_builder": "hair_gan.hair_swap.HairFast",
        "params": model_config.hair_gan_params,
        "weight": {
            "type": "hf_folder",
            "options": {
                "repo_id": "AIRI-Institute/HairFastGAN",
                "repo_type": "model",
                "local_dir": ROOT_DIR,
                "revision": "main",
                "allow_patterns": ["pretrained_models/*"],
            },
        },
    },
)
