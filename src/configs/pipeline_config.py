import os
from yaml import safe_load
from pathlib import Path
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    epoch_num: int
    batch_size: int
    learn_rate: float
    adv_rate: float
    rec_rate: float
    p_rate: float
    h_rate: float
    f_rate: float


class PipelineConfig(BaseModel):
    training: TrainingConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Load config.yaml into a fully validated Pydantic model."""
    path = Path(path)
    with path.open("r") as f:
        data = safe_load(f)

    return PipelineConfig(**data)


work_dir = os.path.dirname(os.path.realpath(__file__))
pipeline_config = load_config(Path(work_dir) / "pipeline_config.yaml")
