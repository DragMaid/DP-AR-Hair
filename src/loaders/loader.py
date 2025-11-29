import torch
from pathlib import Path
from loaders.registry import ModelRegistry
from loaders.downloader import download_weights
from loaders.utils import model_ram_usage


def load_models(name: str, pretrained: bool = False, strict=True):
    """
    Load models from registry and allow convenient load pretrained.
    """
    registry = ModelRegistry.get_registry(name)
    if not registry:
        raise ValueError(f"No model found for name {name}")
    model = registry["model_builder"](**registry["params"].model_dump())
    if pretrained:
        load_weights(model, name, strict=strict)
    return model


def load_weights(model, name: str, strict: bool = True):
    """
    Loads weights into a PyTorch model from registry.
    """
    registry: dict = ModelRegistry.get_registry(name)
    root_path: Path = registry["weight"]["options"]["local_dir"]
    weight_path: Path = root_path / \
        registry["weight"]["options"]["filename"].split("/")[-1]

    if not weight_path.exists():
        download_weights(registry["weight"]["type"],
                         registry["weight"]["options"])

    state = torch.load(
        str(weight_path), map_location="gpu" if torch.cuda.is_available() else "cpu")
    result = model.load_state_dict(state, strict=strict)
    model.eval()

    return {
        "loaded_from": str(weight_path),
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys
    }


if __name__ == "__main__":
    model_names = ["E_C", "E_M", "W", "G"]

    def test_load_models(models):
        for model_name in model_names:
            model = load_models(model_name, pretrained=True)
            print(f"{model_name} ram usage: {model_ram_usage(model):.2f} MB")

    test_load_models(model_names)
