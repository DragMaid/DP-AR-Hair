# Downloader
Location: `src/loaders/downloader.py`

## Overview

The `downloader` module centralizes the logic used by the project to fetch remote weight files and place them into the local `weights/` directory tree. It provides a small mapping of downloader backends and a convenience function `download_weights` used by the higher-level loader utilities.

## Usage

- `download_weights(dtype, options) -> str`:
  - `dtype` (str): key for the downloader backend. Supported values in this repo: `hf_file`, `hf_folder`, `direct_link`.
  - `options` (dict): downloader-specific options. For the `hf_file` backend, the expected keys are:
    - `repo_id` (str): remote repository id on the HuggingFace Hub
    - `repo_type` (str): repository type (e.g. `"space"` or `"model"`)
    - `filename` (str): path to the file inside the repository
    - `local_dir` (Path|str): local directory where files should be downloaded
  - Returns: path to the downloaded file or the return value of the underlying downloader callable.
  - Raises: `ValueError` when the `dtype` is not supported.

## Behavior

- The module maps `dtype` names to small callables in `DOWNLOADER_MAPPER`:
  - `hf_file` → `hf_file_download` (uses `hf_hub_download`)
  - `hf_folder` → `snapshot_download`
  - `direct_link` → `direct_link_download` (uses `urllib.request.urlretrieve`)

- After a file is downloaded by `hf_file_download`, the helper `move_all_files_to_root` from `loaders.utils` is called on `options['local_dir']`. This flattens any subfolders produced by the download into the single `local_dir` root and removes empty subfolders.

## Examples

### hf_file

```python
options = {
    'repo_id': 'KlingTeam/LivePortrait',
    'repo_type': 'space',
    'filename': 'pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth',
    'local_dir': '/path/to/weights'
}
download_weights('hf_file', options)
```

### hf_folder (snapshot of repo folder)

```python
options = {'repo_id': 'AIRI-Institute/HairFastGAN', 'repo_type': 'model', 'local_dir': '/path/to/weights'}
download_weights('hf_folder', options)
```

### direct_link

```python
options = {'link': 'https://example.com/resnet18.pt', 'filename': 'resnet18.pt', 'local_dir': '/path/to/weights'}
download_weights('direct_link', options)
```

## Notes and caveats

- `move_all_files_to_root` will overwrite files in the destination if there are name collisions (older files with the same filename). Ensure `local_dir` points at a unique folder for each download if you want to avoid accidental overwrites.
- The function returns whatever the underlying downloader returns (for `hf_hub_download` that's the downloaded file path as a string).
