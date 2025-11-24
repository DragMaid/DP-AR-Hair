from huggingface_hub import hf_hub_download
from loaders.utils import move_all_files_to_root

DOWNLOADER_MAPPER = {
    "huggingface": lambda repo_id, repo_type, filename, local_dir: hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename, local_dir=local_dir)
}


def download_weights(dtype, options):
    downloader = DOWNLOADER_MAPPER.get(dtype, None)
    if not downloader:
        raise ValueError(f"No downloader found for type {dtype}")
    file_path = downloader(**options)
    move_all_files_to_root(options["local_dir"])
    return file_path
