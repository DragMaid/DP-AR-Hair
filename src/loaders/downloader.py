from urllib.request import urlretrieve
from huggingface_hub import hf_hub_download, snapshot_download
from loaders.utils import move_all_files_to_root


def direct_link_download(link, filename, local_dir):
    if not local_dir or not local_dir.exists() or not link:
        return
    filename = filename if filename else link.split("/")[-1]
    dest = local_dir / filename
    urlretrieve(link, dest)
    return dest


def download_weights(dtype, options):
    downloader = DOWNLOADER_MAPPER.get(dtype, None)
    if not downloader:
        raise ValueError(f"No downloader found for type {dtype}")
    file_path = downloader(**options)
    return file_path


def hf_file_download(**options):
    hf_hub_download(**options)
    move_all_files_to_root(options["local_dir"])


DOWNLOADER_MAPPER = {
    "hf_file": hf_file_download,
    "hf_folder": snapshot_download,
    "direct_link": direct_link_download,
}
