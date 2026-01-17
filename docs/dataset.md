# Dataset module

## Overview
This page documents the dataset helpers used by the project (located under `src/data`). The datasets provide paired driving images (frontal and side views) and a reference image for training and evaluation.

## Supported dataset classes
- `CelebVHQGeneratedDataset` — uses a generated image alongside matching driving pair files.
- `CelebVHQReferenceDataset` — pairs driving images with a random reference image drawn from a folder.
- `_CelebVHQBase` — small base class that provides common loading and transform utilities.

## Common return format
Each dataset returns a dict with three entries: `front`, `side`, and `reference`. Each entry is itself a dict:
- `path`: original file path as a string
- `content`: torch.Tensor with shape [C, H, W], values in [0, 1]

## Constructor arguments (common)
- `driving_dir: str` — directory with driving images (expected filename suffixes: `_frontal.jpg` and `_side.jpg`).
- `reference_dir: str` — directory with reference images (random hairstyles to be put in with driving image to get new generated image)
- `dataset_dir: str` — directory containing all the generated dataset usuable for the training process (expect filename suffixes: `_driving.jpg`, `_reference.jpg` and `_generated.jpg` where driving and reference are the same as `_frontal` and `_side` but renamed to fit the paper terminologies)
- `transform` — optional transform applied to the RGB HxWxC numpy array. If omitted, images are converted to float tensors in [0,1].

## Behavior details
- File name convention: datasets expect driving files organized as `{id}_frontal.jpg` and `{id}_side.jpg` (case-sensitive suffix in code). IDs are derived by removing the trailing underscore and suffix components.
- `CelebVHQGeneratedDataset` scans the `dataset_dir` and matches each generated file to driving pair files with the same ID. It raises `RuntimeError` if no samples are found.
- `CelebVHQReferenceDataset` scans `driving_dir` for IDs that have both front and side images and keeps a list. It also collects all reference image paths up front and raises `RuntimeError` if none are found.

## Image loading and transforms
- Images are loaded with `cv2.imread` (BGR), converted to RGB via `cv2.cvtColor`. If `cv2.imread` returns `None`, a `FileNotFoundError` is raised.
- If `transform` is provided it will be called with an RGB HxWxC numpy array (the repo commonly uses torchvision transforms which expect PIL or numpy input after wrapping in `ToPILImage`). If no transform is provided, images are converted to tensors via `torch.from_numpy(...).permute(2,0,1).float()/255.0`.

## Preload / caching (notes)
Some dataset variants may support preloading driving images into memory or caching reference loads. Benefits:
- Preloading avoids repeated disk access at the cost of increased RAM usage.

## Indexing and randomness
- `__len__` returns the number of discovered samples.
- `__getitem__` returns a random reference image on each call for `CelebVHQReferenceDataset` (so training sees varied references).

## Return example
```python
{
  "front": {"path": ".../123_frontal.jpg", "content": Tensor[C,H,W]},
  "side": {"path": ".../123_side.jpg", "content": Tensor[C,H,W]},
  "reference": {"path": ".../ref_45.jpg", "content": Tensor[C,H,W]},
}
```

## Usage example
Minimal usage without transforms:

```python
from src.data.celebvhq_reference import CelebVHQReferenceDataset
dataset = CelebVHQReferenceDataset(driving_dir="./assets/driving_images",
                                   reference_dir="./assets/reference_images",
                                   transform=None)
sample = dataset[0]
```

With torchvision transforms (resize + ToTensor):

```python
from torchvision import transforms as T
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256,256)),
    T.ToTensor(),
])
```

## Edge cases and tips
- Filenames not following the `_frontal` / `_side` pattern are ignored.
- `cv2.imread` failures raise `FileNotFoundError` — check permissions and paths.
- Using `transform` that expects a PIL image is fine if you first wrap with `T.ToPILImage()`.
- If you need reproducible reference selection, control randomness externally (set seeds) or modify dataset to deterministically sample.

## Testing suggestions
- Create temporary directories with mocked image files (small valid PNGs) and assert the dataset finds pairs correctly.
- Test that the dataset raises `RuntimeError` when no valid pairs or references exist.
- Test transform behavior both with `transform=None` and with torchvision transforms.

## See also
- `src/data/celebvhq_base.py` — base loader utilities.
- `src/data/celebvhq_generated.py` and `src/data/celebvhq_reference.py` — dataset implementations.
