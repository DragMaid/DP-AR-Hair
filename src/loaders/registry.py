from pathlib import Path
from typing import Set, Dict
from uuid import uuid4
from collections import defaultdict
from configs.model_config import model_config
from models.synthesis_decoder import SynthesisDecoder
from face_parsing.models.bisenet import BiSeNet
from live_portrait.models.appearance_feature_extractor import AppearanceFeatureExtractor
from live_portrait.models.motion_extractor import MotionExtractor
from live_portrait.models.warping_network import WarpingNetwork
from live_portrait.models.context_decoder import ContextDecoder
from hair_gan.hair_swap import HairFast

ROOT_DIR = Path(__file__).resolve().parents[2]
WEIGHT_ROOT = ROOT_DIR / "weights"


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
    def get_registry(cls, name: str) -> dict:
        if name not in cls._alias_map.keys():
            raise KeyError(f"Model '{name}' not found in registry.")

        registry = cls._registry.get(cls._alias_map[name], None)
        # Second check just to be sure
        if not registry:
            raise ValueError(f"Expected model registry, got {registry}")
        return registry

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
        "params": model_config.appearance_feature_extractor_params,
        "weight": {
            "type": "huggingface",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",  # Not implemented yet
        "precision": "fp32"       # Not implemented yet
    }
)

ModelRegistry.register(
    {"motion_extractor", "E_M"},
    {
        "model_builder": MotionExtractor,
        "params": model_config.motion_extractor_params,
        "weight": {
            "type": "huggingface",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",  # Not implemented yet
        "precision": "fp32"       # Not implemented yet
    }
)

ModelRegistry.register(
    {"warping_module", "W"},
    {
        "model_builder": WarpingNetwork,
        "params": model_config.warping_module_params,
        "weight": {
            "type": "huggingface",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/warping_module.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",  # Not implemented yet
        "precision": "fp32"       # Not implemented yet
    }
)

ModelRegistry.register(
    {"spade_generator", "context_decoder", "D_C"},
    {
        "model_builder": ContextDecoder,
        "params": model_config.context_decoder_params,
        "weight": {
            "type": "huggingface",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/spade_generator.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",  # Not implemented yet
        "precision": "fp32"       # Not implemented yet
    }
)


ModelRegistry.register(
    {"Hair Mask", "M_C"},
    {
        "model_builder": BiSeNet,
        "params": model_config.face_parsing_params,
        "weight": {
            "type": "direct_link",
            "options": {
                # Will only do ResNet18 for now
                "link": "https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/resnet18.pt",
                "filename": "resnet18.pt",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",  # Not implemented yet
        "precision": "fp32"       # Not implemented yet
    }
)

ModelRegistry.register(
    {"synthesis_decoder", "D_S"},
    {
        "model_builder": SynthesisDecoder,
        "params": model_config.context_decoder_params,
        "weight": {
            "type": "huggingface",
            "options": {
                "repo_id": "KlingTeam/LivePortrait",
                "repo_type": "space",
                "filename": "pretrained_weights/liveportrait/base_models/spade_generator.pth",
                "local_dir": WEIGHT_ROOT,
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",  # Not implemented yet
        "precision": "fp32"       # Not implemented yet
    }
)

ModelRegistry.register(
    {"gan_hair", "IIHT1"},  # NOTE: This model loads itself
    {
        "model_builder": HairFast,
        "params": model_config.hair_gan_params,
        "weight": {
            "type": "huggingface",
            "options": {
                "repo_id": "AIRI-Institute/HairFastGAN",
                "repo_type": "space",
                "filename": "pretrained_models",
                "local_dir": ROOT_DIR / "libs/hair_gan",
            },
        },
        "loader": "pytorch",
        "key_mapper": "default",  # Not implemented yet
        "precision": "fp32"       # Not implemented yet
    }
)

if __name__ == "__main__":
    from loaders.downloader import download_weights
    tmp = ModelRegistry.get_registry("gan_hair")
    download_weights(tmp["weight"]["type"],
                     tmp["weight"]["options"])
