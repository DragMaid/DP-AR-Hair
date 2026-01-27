import torch
from pathlib import Path
from loaders.registry import ModelRegistry
from loaders.downloader import download_weights


def load_models(name: str, pretrained: bool = False,
                strict: bool = True, freeze: bool = False,
                params: dict = None):
    """
    Load models from registry and allow convenient load pretrained.
    """
    registry = ModelRegistry.get_registry(name)
    if not registry:
        raise ValueError(f"No model found for name {name}")
    params = registry["params"].model_dump() if not params else params
    model = registry["model_builder"](**params)

    if pretrained:
        results = load_weights(model, name, strict=strict)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = name not in results["loaded_keys"]

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
        str(weight_path), map_location="cuda" if torch.cuda.is_available() else "cpu")
    result = model.load_state_dict(state, strict=strict)
    model.eval()

    return {
        "loaded_from": str(weight_path),
        "loaded_keys": list(state.keys()),
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys
    }


def load_hfg_generator():
    """Initialize the weights and return the generator instance."""

    name = "IIHT1"
    record = ModelRegistry.get_registry(name)
    w_options = record["weight"]["options"]
    dest = w_options["local_dir"] / \
        w_options["allow_patterns"][0].split("/")[0]

    if not dest.exists():
        download_weights(record["weight"]["type"], w_options)

    # The model load weights by itself so pretrained is False
    return load_models(name, pretrained=False)


if __name__ == "__main__":
    from loaders.utils import model_ram_usage
    model_names = ["E_H", "E_M", "W", "G"]

    def test_load_models(models):
        for model_name in model_names:
            model = load_models(model_name, pretrained=True)
            print(f"{model_name} ram usage: {model_ram_usage(model):.2f} MB")

    test_load_models(model_names)
