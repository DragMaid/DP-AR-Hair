Dataset module (src/data/dataset.py)

Overview
--------
This document describes the `CelebVHQDataset` and supporting dataclasses defined in `src/data/dataset.py`.

The module provides a compact, efficient PyTorch Dataset optimized for a simple file layout used by this project:
- driving images (pairs): `driving_images/{id}_front.jpg` and `driving_images/{id}_side.jpg`
- reference images: arbitrary images under `reference_images/`

It returns a dictionary with three tensors: `front`, `side` and `reference`.

Public classes and dataclasses
------------------------------
- `VideoClip` (dataclass)
  - Simple container used elsewhere in the codebase (clip_id, ytb_id, start_sec, end_sec, bbox).

- `FramePair` (dataclass)
  - Small container describing two selected frames (frontal and side) with their pose and indices.

- `CelebVHQDataset(torch.utils.data.Dataset)`
  - A Dataset that reads front/side image pairs and a random reference image on each sample.

CelebVHQDataset: constructor
----------------------------
Arguments:
- `driving_dir: str` – path to folder containing paired driving images (front/side).
- `reference_dir: str` – path to folder containing reference images.
- `transform` – optional torchvision-like transform applied to RGB numpy arrays.
- `preload: bool` – when True, loads all driving images into memory at construction.
- `cache_size: int` – size of LRU cache used for reference images (default 64).

Behavior and details
--------------------
1. Scanning driving images
   - The dataset scans `driving_dir` for files that end with `_front.(jpg|jpeg|png)` and `_side.(jpg|jpeg|png)` (case-insensitive).
   - Only IDs which have both front and side images are kept. If no valid pairs are found, the constructor raises `RuntimeError`.

2. Reference images
   - The `reference_dir` is scanned for images with suffix `.jpg`, `.jpeg`, `.png` (case-insensitive) and must contain at least one file, otherwise `RuntimeError` is raised.

3. Preload
   - If `preload=True`, driving images are read once during initialization and stored in `self.preloaded_driving` as BGR numpy arrays. When samples are accessed, they are converted to RGB.
   - Preloading may significantly increase memory usage.

4. Reference LRU cache
   - A small LRU cache (via `functools.lru_cache`) is created to speed up reference image loads. The cache expects file path strings and stores raw BGR images returned by `cv2.imread`.

5. __getitem__(idx)
   - Loads front and side images (from the preloaded cache if available, otherwise via `cv2.imread`). The images are converted from BGR to RGB.
   - A random reference image path is chosen and loaded via the LRU cache, then converted to RGB.
   - If `transform` is provided, all three images are passed through it. The transform should accept a HxWxC RGB numpy array (not BGR).
   - If no transform is provided, images are converted to torch tensors with channel-first layout and normalized to [0, 1].

Return value
------------
A dict with keys:
- `'front'`: Tensor[C,H,W]
- `'side'`: Tensor[C,H,W]
- `'reference'`: Tensor[C,H,W]

Usage examples
--------------
Minimal usage without transforms:

    from src.data.dataset import CelebVHQDataset
    dataset = CelebVHQDataset(driving_dir="./assets/driving_images",
                              reference_dir="./assets/reference_images",
                              transform=None)
    sample = dataset[0]
    front = sample['front']  # torch.Tensor

With torchvision transforms (example resize + ToTensor):

    from torchvision import transforms as T
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512,512)),
        T.ToTensor(),
    ])
    dataset = CelebVHQDataset(..., transform=transform)

Important notes and edge cases
-----------------------------
- File name conventions: the dataset relies on the `_front` and `_side` suffixes. Filenames that don't follow this convention will be ignored.
- Image loading failures: `cv2.imread` returning `None` raises `FileNotFoundError` in `_load_image` and also inside the reference LRU loader. Ensure paths are correct and files are readable.
- Preloading: If images are preloaded, they are stored as BGR arrays. The dataset converts them to RGB on access. This doubles memory pressure briefly during conversion.
- Random reference choice: `__getitem__` picks a random reference image for every call; when used with a DataLoader and multiple workers the randomness is per-worker.

Performance tips
----------------
- If you have plenty of RAM, enable `preload=True` to avoid repeated disk reads of driving images.
- Keep `cache_size` tuned to the number of distinct reference images you expect to access; a larger cache will use more memory but reduce disk I/O.
- Use pinned memory and appropriate DataLoader `num_workers` for faster host-to-device transfers.

Testing suggestions
-------------------
- Unit tests should verify scanning behavior: create temporary files with proper suffixes and ensure pairs are detected correctly.
- Test that reference images are chosen randomly and cached by mocking `cv2.imread`.
- Test `preload=True` branch and ensure outputs are valid tensors and properly normalized.

See also
--------
- `src/data/preprocess.py` for video processing and frame extraction tools used upstream in dataset preparation.

Changelog / Notes
-----------------
This documentation is derived from the inline docstrings and the original `src/data/dataset.py` implementation. It focuses on public behavior and recommended usage patterns.

