# Context Encoder (`context_decoder.py`)

## Overview
The `ContextEncoder` module implements a 2D encoder that extracts spatial context features for downstream decoders (for example SPADE / MSG decoders). It is a pure 2D encoder (no volumetric reshape) and projects the final feature map to `out_channels` channels suitable for the decoder.

Typical output shape shown in the original project is `B × 256 × 64 × 64` when using the project's default settings.

## Class: `ContextEncoder`
Brief description: a configurable convolutional encoder composed of an initial convolutional block, a sequence of downsampling blocks, and a small projection head.

### Constructor signature
`ContextEncoder(image_channel, block_expansion, num_down_blocks, max_features, out_channels=256)`

### Parameters
- `image_channel` (int): Number of channels in the input image (for example `3` for RGB).
- `block_expansion` (int): Base number of feature maps for the first block; higher values expand feature width exponentially across downsampling blocks.
- `num_down_blocks` (int): Number of downsampling blocks to apply after the initial block.
- `max_features` (int): Upper bound on feature channel count to avoid unbounded growth.
- `out_channels` (int, optional): Number of output channels from the projection head. Default: `256`.

### Attributes (important parts)
- `first`: initial `SameBlock2d` convolutional block (preserves spatial size).
- `down_blocks`: `nn.ModuleList` of `DownBlock2d` instances used to reduce spatial resolution and increase channels.
- `proj`: small `nn.Sequential` projection (1x1 conv -> LeakyReLU -> 1x1 conv) that maps to `out_channels`.

### Forward behavior
- Input: a 4-D tensor `x` with shape `(B, image_channel, H, W)`.
- The module applies `first`, then each block in `down_blocks`, and finally `proj`.
- Output: a 4-D tensor `f_c` with shape `(B, out_channels, H_out, W_out)`.
  - `H_out` and `W_out` depend on `num_down_blocks` and the downsample behavior of `DownBlock2d`. With the project's standard inputs (e.g., input `256×256` and configured downsampling), the output is typically `64×64`.

## Example usage
```python
from torch import nn
from live_portrait.models.context_decoder import ContextEncoder

# instantiate (example values)
encoder = ContextEncoder(
    image_channel=3,
    block_expansion=32,
    num_down_blocks=3,
    max_features=256,
    out_channels=256
)

# forward pass
x = torch.randn(2, 3, 256, 256)  # batch of 2 RGB images
f_c = encoder(x)  # f_c shape: (2, 256, H_out, W_out)
