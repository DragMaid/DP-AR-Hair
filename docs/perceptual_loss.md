Perceptual Loss (VGG-based)

Summary
-------
This module provides a perceptual loss that compares intermediate feature
maps from a pretrained VGG-16 network between a generated image and a target
image. It's commonly used to encourage perceptual similarity (texture and
structure) rather than only pixel-wise similarity.

Source
------
`src/losses/perceptual_loss.py`

Public API
----------
- class PerceptualLoss(layers=[2, 7, 14, 21])

Description
-----------
`PerceptualLoss` constructs a VGG-16 feature extractor (the `features` part of
`torchvision.models.vgg16`) and uses selected layer outputs to compute an
MSE loss between the generated and target images' features.

Constructor arguments
---------------------
- `layers` (list[int], optional): indices of VGG feature layers to use when
  computing the perceptual loss. Default: [2, 7, 14, 21].

Forward
-------
- Inputs: `generated` and `target` — tensors expected to be normalized and of
  shape (B, 3, H, W).
- Output: scalar tensor representing the sum of MSE losses across the
  specified VGG layers.

Example usage
-------------
```python
import torch
from src.losses.perceptual_loss import PerceptualLoss

loss_fn = PerceptualLoss()
g = torch.randn(1, 3, 224, 224)
t = torch.randn(1, 3, 224, 224)
loss = loss_fn(g, t)
```

Notes and Known Issues
----------------------
While documenting the implementation I noticed a couple of bugs in the source
file that should be fixed for the module to work correctly:

1. Typo in `for param in self.vgg.parameers():` — should be `parameters()`.
2. The code uses `nn.function.mse_loss` but should instead use
   `nn.functional.mse_loss` or `torch.nn.functional.mse_loss`.

Suggested fix (summary):
```python
for param in self.vgg.parameters():
    param.requires_grad = False

import torch.nn.functional as F
# ...
loss += F.mse_loss(gen_features, target_features)
```

Because this module loads a pretrained VGG-16, running unit tests that import
this file will attempt to download weights unless local cached weights are
available. During CI it's common to mock or monkeypatch the model-loading to
avoid network calls.

References
----------
- "Perceptual Losses for Image Transformation" (Johnson et al.)
- torchvision.models.vgg16

