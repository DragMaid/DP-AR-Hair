import torch
import shutil
from pathlib import Path
from loaders.loader import load_models, ModelRegistry
from loaders.downloader import download_weights


def model_ram_usage(model: torch.nn.Module, dtype=torch.float32):
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()

    # Size in bytes
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    mem_bytes = total_params * bytes_per_param
    mem_mb = mem_bytes / (1024 ** 2)
    return mem_mb


def move_all_files_to_root(root_dir: str | Path) -> None:
    """
    Move all files inside root_dir (including subfolders)
    into the root_dir itself.

    After moving, empty subfolders are automatically removed.
    """
    root_dir = Path(root_dir)

    # Move files from subfolders to root
    for path in root_dir.rglob("*"):
        if path.is_file() and path.parent != root_dir:
            dest = root_dir / path.name
            # Handle name collision by overwriting
            if dest.exists():
                dest.unlink()
            shutil.move(str(path), str(dest))

    # Remove empty subfolders
    for folder in sorted(root_dir.rglob("*"), reverse=True):
        if folder.is_dir():
            try:
                folder.rmdir()
            except OSError:
                pass


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
