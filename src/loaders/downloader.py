from urllib.request import urlretrieve
from huggingface_hub import hf_hub_download
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
    move_all_files_to_root(options["local_dir"])
    return file_path


DOWNLOADER_MAPPER = {
    # TODO: maybe rewrite this to be more readable
    "huggingface": hf_hub_download,
    "direct_link": direct_link_download,
}
