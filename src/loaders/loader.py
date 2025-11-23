# weights/loader.py

import torch
from pathlib import Path
from loaders.registry import WeightRegistry


def load_weights(model, name: str, strict: bool = True):
    """
    Loads weights into a PyTorch model from registry.
    """
    weight_path: Path = WeightRegistry.get_path(name)

    if not weight_path.exists():
        # TODO: replace this with the download instead
        raise FileNotFoundError(
            f"Weight file '{weight_path}' does not exist. Did you download it?"
        )

    state = torch.load(str(weight_path), map_location="cpu")

    if "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=strict)

    return {
        "loaded_from": str(weight_path),
        "missing_keys": missing,
        "unexpected_keys": unexpected
    }
