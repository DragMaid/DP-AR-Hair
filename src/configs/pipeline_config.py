import os
from yaml import safe_load
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple


class StablizersConfig(BaseModel):
    mask_jitter_prob: float
    image_aug_prob: float


class LossConfig(BaseModel):
    adv_rate: float
    rec_rate: float
    p_rate: float
    h_rate: float
    f_rate: float


class GeneratorConfig(BaseModel):
    learn_rate: float
    betas: Tuple[float, float]


class DiscriminatorConfig(BaseModel):
    learn_rate: float
    betas: Tuple[float, float]


class LoggingConfig(BaseModel):
    grad_contrib_interval: int
    param_norm_interval: int
    param_ratio_interval: int
    param_dist_interval: int
    param_hist_interval: int
    ema_update_interval: int
    output_save_interval: int


class TrainingConfig(BaseModel):
    epoch_num: int
    batch_size: int
    mini_batch_size: int
    loss: LossConfig
    discriminator: DiscriminatorConfig
    generator: GeneratorConfig
    stablizers: StablizersConfig
    save_dir: str
    artifact_dir: str
    epochs_till_save: int
    steps_till_save: int
    log_config: LoggingConfig


class InferenceConfig(BaseModel):
    video_path: str
    reference_path: str
    output_path: str
    checkpoint_path: str
    batch_size: int
    frame_size: int
    fps: int


class DatasetConfig(BaseModel):
    dataset_dir: str
    num_workers: int
    device: int


# Config for dataset generation
class GenerationConfig(BaseModel):
    reference_dir: str
    driving_dir: str
    cache_path: str


class PipelineConfig(BaseModel):
    training: TrainingConfig
    dataset: DatasetConfig
    generation: GenerationConfig
    inference: InferenceConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Load config.yaml into a fully validated Pydantic model."""
    path = Path(path)
    with path.open("r") as f:
        data = safe_load(f)

    return PipelineConfig(**data)


work_dir = os.path.dirname(os.path.realpath(__file__))
pipeline_config = load_config(Path(work_dir) / "pipeline_config.yaml")
