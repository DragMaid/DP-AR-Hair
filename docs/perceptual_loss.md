# Perceptual Loss (VGG-based)
Location: `src/losses/perceptual_loss.py`

## Overview
This module provides a perceptual loss that compares intermediate feature
maps from a pretrained VGG-16 network between a generated image and a target
image. It's commonly used to encourage perceptual similarity (texture and
structure) rather than only pixel-wise similarity.

## Description
`PerceptualLoss` constructs a VGG-16 feature extractor (the `features` part of
`torchvision.models.vgg16`) and uses selected layer outputs to compute an
MSE loss between the generated and target images' features.

## Constructor arguments
- `layers` (list[int], optional): indices of VGG feature layers to use when
  computing the perceptual loss. Default: [2, 7, 14, 21].

## Forward
- Inputs: `generated` and `target` — tensors expected to be normalized and of
  shape (B, 3, H, W).
- Output: scalar tensor representing the sum of MSE losses across the
  specified VGG layers.

## Example usage
```python
import torch
from src.losses.perceptual_loss import PerceptualLoss

loss_fn = PerceptualLoss()
g = torch.randn(1, 3, 224, 224)
t = torch.randn(1, 3, 224, 224)
loss = loss_fn(g, t)
```

## Notes
Because this module loads a pretrained VGG-16, running unit tests that import
this file will attempt to download weights unless local cached weights are
available. During CI it's common to mock or monkeypatch the model-loading to
avoid network calls.

## References
- "Perceptual Losses for Image Transformation" (Johnson et al.)
- torchvision.models.vgg16
