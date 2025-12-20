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


class ContextEncoderParams(BaseModel):
    image_channel: int
    block_expansion: int
    num_down_blocks: int
    max_features: int
    out_channels: int


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


class SynthesisDecoderParams(BaseModel):
    upscale: int
    block_expansion: int
    max_features: int
    num_down_blocks: int


class FaceParsingParams(BaseModel):
    num_classes: int
    backbone_name: str


class HairGanParams(BaseModel):
    save_all_dir: str
    size: int
    ckpt: str
    channel_multiplier: int
    latent: int
    n_mlp: int
    device: str
    batch_size: int
    save_all: bool
    mixing: float
    smooth: int
    rotate_checkpoint: str
    blending_checkpoint: str
    pp_checkpoint: str


# ---------------------------
# Top-level config
# ---------------------------

class ModelConfig(BaseModel):
    appearance_feature_extractor_params: AppearanceFeatureExtractorParams
    context_encoder_params: ContextEncoderParams
    motion_extractor_params: MotionExtractorParams
    warping_module_params: WarpingModuleParams
    context_decoder_params: ContextDecoderParams
    face_parsing_params: FaceParsingParams
    hair_gan_params: HairGanParams
    synthesis_decoder_params: SynthesisDecoderParams


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
model_config = load_config(Path(work_dir) / "model_config.yaml")
