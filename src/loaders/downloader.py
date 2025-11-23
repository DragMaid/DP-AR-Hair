from huggingface_hub import hf_hub_download
from pathlib import Path

WEIGHT_ROOT = Path(__file__).resolve().parent / "files"
WEIGHT_ROOT.mkdir(exist_ok=True, parents=True)


def download_liveportrait_weights(file, repo, target_dir):
    file_path = hf_hub_download(
        repo_id="KlingTeam/LivePortrait",
        repo_type="space",
        filename="pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    return file_path
