# Model Registry

## Purpose

The `ModelRegistry` (`src/loaders/registry.py`) is a lightweight registry that centralizes model constructors, default parameters, and weight download metadata for the project. It serves as the single source of truth mapping small string aliases (like `E_M`, `W`, `G`) to concrete model builders and their associated weight locations.

## Key concepts

- Each registered entry contains the following fields:
  - `model_builder`: a callable/class that constructs the model (e.g., `MotionExtractor`)
  - `params`: an OmegaConf/structured config object that provides model constructor kwargs (via `model_dump()`)
  - `weight`: information for how to download and where to place pretrained weights
  - `loader`: loader type (currently `pytorch`)
  - `key_mapper`, `precision`: placeholder fields for future extensions

- Registry internals:
  - `_registry` stores entries keyed by a generated UUID
  - `_alias_map` maps human-friendly aliases (like `"E_M"`) to those UUIDs

## Public API

- `ModelRegistry.register(names: Set[str], data: Dict)` (class method)
  - Registers a new model with one or more canonical aliases.
  - `names` is a set of strings (aliases). Each alias will point to the same registry entry.
  - `data` is a dictionary with the fields described above.

- `ModelRegistry.get_registry(name: str) -> dict` (class method)
  - Returns the registry entry for `name` (alias-based lookup).
  - Raises `KeyError` when the alias is not found.
  - Raises `ValueError` if the registry entry is missing or malformed.

- `ModelRegistry.list()` (class method)
  - Returns a compact representation of aliases and their registry entries.

## How models are registered in this project

The repository pre-registers the following models (aliases shown in braces):

- `AppearanceFeatureExtractor` ({`appearance_feature_extractor`, `E_H`, `E_C`})
  - `model_builder`: `AppearanceFeatureExtractor`
  - `params`: `model_config.appearance_feature_extractor_params`
  - `weight`: HuggingFace Space `KlingTeam/LivePortrait` `appearance_feature_extractor.pth`

- `MotionExtractor` ({`motion_extractor`, `E_M`})
  - `model_builder`: `MotionExtractor`
  - `params`: `model_config.motion_extractor_params`
  - `weight`: HuggingFace Space `KlingTeam/LivePortrait` `motion_extractor.pth`

- `WarpingNetwork` ({`warping_module`, `W`})
  - `model_builder`: `WarpingNetwork`
  - `params`: `model_config.warping_module_params`
  - `weight`: HuggingFace Space `KlingTeam/LivePortrait` `warping_module.pth`

- `ContextDecoder`/`SPADE` generator ({`spade_generator`, `context_decoder`, `D_C`, `G`})
  - `model_builder`: `ContextDecoder`
  - `params`: `model_config.context_decoder_params`
  - `weight`: HuggingFace Space `KlingTeam/LivePortrait` `spade_generator.pth`

- `BiSeNet` face parsing model ({`Hair Mask`, `M_C`})
  - `model_builder`: `BiSeNet`
  - `params`: `model_config.bi_se_net_params`
  - `weight`: HuggingFace Space `KlingTeam/LivePortrait` `bi_se_net.pth`

- `SynthesisDecoder` ({`synthesis_decoder`, `D_S`})
  - `model_builder`: `SynthesisDecoder`
  - `params`: `model_config.synthesis_decoder_params`
  - `weight`: HuggingFace Space `KlingTeam/LivePortrait` `synthesis_decoder.pth`

- `HairFast` GAN ({`gan_hair`, `IIHT1`})
  - `model_builder`: `HairFast`
  - `params`: `model_config.hair_gan_params`
  - `weight`: HuggingFace Space `KlingTeam/LivePortrait` `hair_gan.pth`

Each registry entry uses a common `WEIGHT_ROOT` on disk (project `weights/` root) where downloaded weights are stored. The repository sets this default via:

```python
from pathlib import Path
WEIGHT_ROOT = Path(__file__).resolve().parents[2] / "weights"
```

## Example usage

```python
from loaders.registry import ModelRegistry

# Inspect list of registered models
print(ModelRegistry.list())

# Lookup a registry entry
entry = ModelRegistry.get_registry("E_M")
print(entry["model_builder"])  # MotionExtractor class

# Unknown alias will raise
try:
    ModelRegistry.get_registry('not_a_model')
except KeyError as e:
    print('Expected error:', e)
```

## Extending the registry

- To register new models, call `ModelRegistry.register(names, data)` where `names` is a set of alias strings and `data` follows the same shape as existing entries.
- Keep registry entries small and avoid large objects as values — prefer references (classes, config objects, file paths).

## Notes & limitations

- Alias lookup is case-sensitive. Use the exact alias strings as registered.
- `ModelRegistry` does not currently provide unregister/overwrite semantics — registering duplicate aliases will overwrite the internal `_alias_map`, but the `_registry` stores entries by UUID.
- `key_mapper` and `precision` fields are placeholders and not used by core loader logic yet.
