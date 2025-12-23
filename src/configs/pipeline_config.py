import os
from yaml import safe_load
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple


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


class TrainingConfig(BaseModel):
    epoch_num: int
    batch_size: int
    loss: LossConfig
    discriminator: DiscriminatorConfig
    generator: GeneratorConfig
    save_dir: str
    epochs_till_save: int
    steps_till_save: int


class DatasetConfig(BaseModel):
    reference_dir: str
    driving_dir: str
    generated_dir: str
    num_workers: int
    device: int


class PipelineConfig(BaseModel):
    training: TrainingConfig
    dataset: DatasetConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Load config.yaml into a fully validated Pydantic model."""
    path = Path(path)
    with path.open("r") as f:
        data = safe_load(f)

    return PipelineConfig(**data)


work_dir = os.path.dirname(os.path.realpath(__file__))
pipeline_config = load_config(Path(work_dir) / "pipeline_config.yaml")
