# Context Decoder

## Overview

The `ContextDecoder` is a generative decoder module that synthesizes visual content from latent features conditioned on semantic context. It serves as the synthesis network in the Hair-Shifter pipeline, transforming dense feature representations into high-resolution RGB images using spatially-adaptive normalization (SPADE) and multi-scale upsampling.

## Architecture

### Input
- **feature**: Input feature tensor of shape `(B, C, H, W)` where:
  - B = batch size
  - C = feature channels (typically 256)
  - H, W = spatial dimensions (typically 64x64)
  - Used as both the feature to decode and the segmentation condition

### Output
- **x**: RGB image tensor of shape `(B, 3, H', W')` where:
  - Values are in range [0, 1] (after sigmoid)
  - H', W' = upsampled spatial dimensions (typically 256x256)

## Component Breakdown

### 1. Feature Preparation

```python
self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
```

Initial channel expansion:
- Input: `(B, 256, 64, 64)`
- Output: `(B, 512, 64, 64)`
- 3×3 convolution with padding to maintain spatial dimensions

### 2. Middle Processing Layers

```python
self.G_middle_0 = SPADEResnetBlock(...)
self.G_middle_1 = SPADEResnetBlock(...)
self.G_middle_2 = SPADEResnetBlock(...)
self.G_middle_3 = SPADEResnetBlock(...)
self.G_middle_4 = SPADEResnetBlock(...)
self.G_middle_5 = SPADEResnetBlock(...)
```

Six SPADE Residual Blocks:
- Each maintains channel dimension: `(B, 512, 64, 64)`
- Conditioned on input feature (acts as segmentation)
- Uses spectral normalization for training stability
- Progressive feature refinement

### 3. Upsampling Layers

```python
self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, ...)
self.up_1 = SPADEResnetBlock(input_channels, out_channels, ...)
self.up = nn.Upsample(scale_factor=2)
```

Upsampling sequence:
- **First Upsample**: `(B, 512, 64, 64)` → `(B, 512, 128, 128)`
- **First SPADE Block**: `(B, 512, 128, 128)` → `(B, 256, 128, 128)`
- **Second Upsample**: `(B, 256, 128, 128)` → `(B, 256, 256, 256)`
- **Second SPADE Block**: `(B, 256, 256, 256)` → `(B, 64, 256, 256)`

### 4. Output Generation

```python
if self.upscale is None or self.upscale <= 1:
    self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
else:
    self.conv_img = nn.Sequential(
        nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
        nn.PixelShuffle(upscale_factor=2)
    )
```

Image generation:
- **Standard Mode**: 3×3 convolution to 3 channels
- **Upscale Mode**: PixelShuffle for additional 2× upsampling
- Final activation: Sigmoid to [0, 1] range

## Forward Pass

```
Input Feature (B, 256, 64, 64)
    ↓
FC: Conv2d (expand channels)
    → (B, 512, 64, 64)
    ↓
Middle Blocks (×6 SPADE ResBlocks)
    → Conditioned on input feature
    → (B, 512, 64, 64) [maintained]
    ↓
Upsample ×1 (scale_factor=2)
    → (B, 512, 128, 128)
    ↓
up_0: SPADE ResBlock (512 → 256)
    → (B, 256, 128, 128)
    ↓
Upsample ×1 (scale_factor=2)
    → (B, 256, 256, 256)
    ↓
up_1: SPADE ResBlock (256 → 64)
    → (B, 64, 256, 256)
    ↓
LeakyReLU Activation
    ↓
conv_img: Conv2d (64 → 3) or PixelShuffle
    → (B, 3, 256×upscale, 256×upscale)
    ↓
Sigmoid Activation
    ↓
Output Image (B, 3, H', W') ∈ [0, 1]
```

## Parameters

```python
ContextDecoder(
    upscale=1,              # Additional upscale factor (0, 1, 2, etc.)
    max_features=256,       # Maximum feature channels
    block_expansion=64,     # Base channel expansion
    out_channels=64,        # Channels before final projection
    num_down_blocks=2       # Number of encoding downsampling blocks
)
```

### Key Parameters Explained

- **upscale**: 
  - `1` or `None`: Standard output resolution
  - `2`: 2× additional upsampling via PixelShuffle
  
- **num_down_blocks**: 
  - Determines input_channels: `min(max_features, block_expansion * 2^(num_down_blocks+1))`
  - Default: 256 channels

## SPADE Conditioning

All SPADE blocks are conditioned on the input feature:

```python
x = self.G_middle_0(x, seg)  # seg = input feature
```

This means:
- Semantic information from input feature guides generation
- Enables semantic-aware synthesis
- Preserves structural information during upsampling
- Condition is consistent throughout the network

## Activation Functions

```python
def actvn(self, x):
    return F.leaky_relu(x, 2e-1)
```

- **Middle Blocks**: LeakyReLU with slope 0.2
- **Pre-image Projection**: LeakyReLU with slope 0.2
- **Output**: Sigmoid for image range [0, 1]

## Design Rationale

### Multiple Middle Blocks
- 6 SPADE blocks enable deep processing
- Allows complex feature transformations
- Maintains spatial resolution for fine details
- Spectral normalization ensures stable training

### Progressive Upsampling
- Two-stage upsampling (2× each)
- Generates from low to high resolution
- Reduces computational cost
- Maintains visual quality through SPADE conditioning

### Semantic Conditioning
- Input feature acts as both content and segmentation
- Enables content-preserving upsampling
- SPADE uses feature information for adaptive normalization
- Improves visual consistency

## Usage Example

```python
from models.context_decoder import ContextDecoder
import torch

# Initialize decoder
decoder = ContextDecoder(
    upscale=1,
    max_features=256,
    block_expansion=64,
    out_channels=64,
    num_down_blocks=2
)

# Decode feature to image
features = torch.randn(4, 256, 64, 64)
output_image = decoder(features)
print(output_image.shape)  # (4, 3, 256, 256)
print(output_image.min(), output_image.max())  # ~0.0 to ~1.0
```

### With Upscaling

```python
# Initialize with 2× upscaling
decoder = ContextDecoder(
    upscale=2,
    num_down_blocks=2
)

features = torch.randn(4, 256, 64, 64)
output_image = decoder(features)
print(output_image.shape)  # (4, 3, 512, 512) - 2× upscaled
```

## Integration with Pipeline

```
Source Image
    ↓
AppearanceFeatureExtractor
    ↓ features (B, 256, 64, 64)
WarpingNetwork (motion adaptation)
    ↓ warped_features (B, 256, 64, 64)
ContextDecoder
    ↓
Output Image (B, 3, 256, 256)
```

## Output Characteristics

- **Range**: [0, 1] due to sigmoid activation
- **Resolution**: 256×256 (or higher with upscale)
- **Color Space**: RGB
- **Differentiable**: All operations support gradient computation
- **Batch Processing**: Efficient batch operations

## Key Characteristics

1. **SPADE-Based**: Semantic-aware normalization for better detail preservation
2. **Multi-Scale Processing**: Progressive upsampling from 64×64 to 256×256
3. **Deep Architecture**: 6 middle blocks enable complex transformations
4. **Spectral Normalization**: Training stability through Lipschitz constraint
5. **Flexible Upscaling**: Optional pixel-shuffle for higher resolution

## Performance Considerations

- **Computational Cost**:
  - Middle blocks: Largest cost
  - Upsampling: ~20% of total
  - Image projection: ~5% of total
  
- **Memory Usage**:
  - Peak memory at 256×256 spatial dimensions
  - Batch processing increases linearly
  
- **Inference Speed**:
  - Optimized for GPU
  - Suitable for real-time applications (batch size dependent)

## Related Components

- **SPADEResnetBlock**: Conditional normalization with residual connections
- **SPADE**: Spatially-adaptive normalization layer
- **PixelShuffle**: Sub-pixel convolution for upsampling
- **Upsample**: PyTorch's upsampling module

## Advanced Usage

### Custom Feature Input
```python
# Use features from other sources
custom_features = extract_custom_features(image)  # (B, 256, 64, 64)
output = decoder(custom_features)
```

### Multi-Scale Output
```python
# For progressive training or multi-scale losses
# Can extract intermediate outputs at different resolutions
```

## Notes

- Input feature must have channels matching calculated input_channels
- All computations are differentiable for end-to-end training
- SPADE conditioning improves quality for semantic image synthesis
- PixelShuffle upscaling preferred over simple interpolation for quality
- Works best when input features contain semantic information

