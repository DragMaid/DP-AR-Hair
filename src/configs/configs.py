import os
from yaml import safe_load
from pathlib import Path
from pydantic import BaseModel


# ---------------------------
# Sub-models
# ---------------------------

class AppearanceFeatureExtractorParams(BaseModel):
    image_channel: int
    block_expansion: int
    num_down_blocks: int
    max_features: int
    reshape_channel: int
    reshape_depth: int
    num_resblocks: int


class MotionExtractorParams(BaseModel):
    num_kp: int


class DenseMotionParams(BaseModel):
    block_expansion: int
    max_features: int
    num_blocks: int
    reshape_depth: int
    compress: int


class WarpingModuleParams(BaseModel):
    num_kp: int
    block_expansion: int
    max_features: int
    num_down_blocks: int
    reshape_channel: int
    estimate_occlusion_map: bool
    dense_motion_params: DenseMotionParams


class ContextDecoderParams(BaseModel):
    upscale: int
    block_expansion: int
    max_features: int
    num_down_blocks: int


# ---------------------------
# Top-level config
# ---------------------------

class ModelConfig(BaseModel):
    appearance_feature_extractor_params: AppearanceFeatureExtractorParams
    motion_extractor_params: MotionExtractorParams
    warping_module_params: WarpingModuleParams
    context_decoder_params: ContextDecoderParams


# ---------------------------
# Loader
# ---------------------------

def load_config(path: str | Path) -> ModelConfig:
    """Load config.yaml into a fully validated Pydantic model."""
    path = Path(path)
    with path.open("r") as f:
        data = safe_load(f)

    return ModelConfig(**data)


work_dir = os.path.dirname(os.path.realpath(__file__))
config = load_config(Path(work_dir) / "configs.yaml")
