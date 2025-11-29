# Loader

## Purpose

The `loader` module (`src/loaders/loader.py`) provides convenient, high-level functions to instantiate models registered in the local `ModelRegistry` and optionally load pretrained weights into them. It uses the registry to look up a model builder and parameter configuration, creates the model, and can fetch and load weights when requested.

## Public API

- `load_models(name: str, pretrained: bool = False)`
  - `name` (str): alias or key for a registered model (see `ModelRegistry` in `registry.py`).
  - `pretrained` (bool): if True, `load_weights` is invoked after model construction.
  - Returns: a PyTorch `nn.Module` instance built using the registered `model_builder` and `params`.
  - Raises: `ValueError` if the registry does not contain the requested model.

- `load_weights(model, name: str, strict: bool = True) -> dict`
  - `model`: an instantiated PyTorch model (the target to load weights into)
  - `name` (str): registry key used to find weight configuration (local path, remote source, etc.)
  - `strict` (bool): forwarded to `model.load_state_dict`; determines strictness of loading
  - Returns: dictionary with keys:
    - `loaded_from` (str): the path to the weights used
    - `missing_keys` (list): keys that were missing in the state dict
    - `unexpected_keys` (list): keys found in the checkpoint that do not map to model params

## Behavior and details

- `load_models`:
  - Retrieves registry entry using `ModelRegistry.get_registry(name)`.
  - Constructs the model via `registry['model_builder'](**registry['params'].model_dump())`.
  - If `pretrained` is True, calls `load_weights(model, name)` to ensure weights are present and loaded.

- `load_weights`:
  - Uses the registry to determine the `local_dir` and expected filename for the weights.
  - If the final `weight_path` does not exist locally, it calls `download_weights(...)` (from `loaders.downloader`) to fetch the file.
  - Loads the weights with `torch.load(..., map_location='cpu')` and applies them to `model` via `model.load_state_dict(state, strict=strict)`.
  - Sets the model to evaluation mode (`model.eval()`) before returning a small diagnostics dict describing loaded/missing/unexpected keys.

## Example

```python
from loaders.loader import load_models

# Construct the motion extractor without loading pretrained weights
motion = load_models("motion_extractor", pretrained=False)

# Construct and load pretrained weights for a registered model
appearance = load_models("appearance_feature_extractor", pretrained=True)
```

## CLI test helper

The module contains a small `__main__` block that demonstrates loading multiple registered models and prints their RAM usage (via `loaders.utils.model_ram_usage`). This is convenient for local smoke tests but not intended for production use.

## Notes and caveats

- `load_weights` assumes the checkpoint file contains a raw state dictionary that is directly compatible with `load_state_dict`. If the repository provides a wrapper (e.g. a dictionary with a `'model'` key), the loader adapts by extracting that key when present.
- The function currently uses CPU `map_location` to avoid CUDA device issues during unit tests and CI; this decision keeps loading deterministic in mixed-device environments.
- The registry entries contain the `weight` configuration used to locate and download weight files — see `src/loaders/registry.py` for how models are registered and where default `local_dir` paths point to.
- The downloader backend types used by the project are: `hf_file`, `hf_folder`, and `direct_link` — these map to the callables in `loaders.downloader.DOWNLOADER_MAPPER`.
