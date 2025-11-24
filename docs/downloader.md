# Downloader

## Purpose

The `downloader` module (source: `src/loaders/downloader.py`) centralizes the logic used by the project to fetch remote weight files and place them into the local `weights/` directory tree. It provides a small mapping of downloader backends and a convenience function `download_weights` used by the higher-level loader utilities.

## Public API

- `download_weights(dtype, options) -> str`:
  - `dtype` (str): key for the downloader backend. Currently supported values: `"huggingface"`.
  - `options` (dict): downloader-specific options. For the `huggingface` backend, the expected keys are:
    - `repo_id` (str): remote repository id on the HuggingFace Hub
    - `repo_type` (str): repository type (e.g. `"space"` or `"model"`)
    - `filename` (str): path to the file inside the repository
    - `local_dir` (str|Path): local directory where files should be downloaded
  - Returns: path to the downloaded file (string returned by `hf_hub_download` used internally).
  - Raises: `ValueError` when the `dtype` is not supported.

## Behavior

- The module maps `dtype` names to small callables in `DOWNLOADER_MAPPER`. The current default mapping uses `huggingface_hub.hf_hub_download`.
- After a file is downloaded, the helper `move_all_files_to_root` from `loaders.utils` is called on `options['local_dir']`. This flattens any subfolders produced by the download into the single `local_dir` root and removes empty subfolders.

## Example

```python
from loaders.downloader import download_weights

options = {
    "repo_id": "KlingTeam/LivePortrait",
    "repo_type": "space",
    "filename": "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
    "local_dir": "/path/to/weights"
}

file_path = download_weights("huggingface", options)
print("Downloaded:", file_path)
```

## Notes and caveats

- The module intentionally keeps the mapping small so adding new sources (S3, HTTP, etc.) is straightforward: add an entry to `DOWNLOADER_MAPPER` that accepts `(repo_id, repo_type, filename, local_dir)` semantics (or adapt the lambda signature where needed).
- `move_all_files_to_root` will overwrite files in the destination if there are name collisions (older files with the same filename). Ensure `local_dir` points at a unique folder for each download if you want to avoid accidental overwrites.
- The function returns whatever the underlying downloader returns (for `hf_hub_download` that's the downloaded file path as a string).

