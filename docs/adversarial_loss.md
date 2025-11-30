Adversarial Loss (PatchGAN Discriminator)

Summary
-------
This module provides a PatchGAN-style discriminator implemented as the
`PatchGANDiscriminator` class. PatchGAN discriminators operate on image
patches and are commonly used in image-to-image translation and GAN-based
architectures where local realism is important.

Source
------
`src/losses/adversarial_loss.py`

Public API
----------
- class PatchGANDiscriminator(n_in_channels, n_filters=64, n_layers=3)

Description
-----------
`PatchGANDiscriminator` builds a convolutional neural network that downsamples
the input image through several strided convolutions and produces a single-
channel output map of discriminator scores.

Constructor arguments
---------------------
- `n_in_channels` (int): Number of input channels for the images (e.g. 3 for RGB).
- `n_filters` (int, optional): Number of filters in the first convolutional
  layer. Default: 64.
- `n_layers` (int, optional): Number of downsampling layers before the final
  convolution. Default: 3.

Forward
-------
- Input: tensor of shape (B, C, H, W)
- Output: tensor of shape (B, 1, H_out, W_out) where H_out/W_out depends on
  the number of strides and kernel sizes used by the network.

Example usage
-------------
```python
import torch
from src.losses.adversarial_loss import PatchGANDiscriminator

D = PatchGANDiscriminator(n_in_channels=3, n_filters=64, n_layers=3)
input = torch.randn(2, 3, 256, 256)  # batch of 2 RGB images
output = D(input)  # shape (2, 1, H', W')
```

Notes
-----
- The output is a map of patch-wise discriminator scores (not a single scalar).
- This discriminator uses LeakyReLU activations and BatchNorm2d between
  intermediate conv layers.
- The kernel size and padding are fixed in the implementation (kernel=4, pad=1).

References
----------
PatchGAN discriminator concept: "Image-to-Image Translation with Conditional
Adversarial Networks" (Isola et al.).

