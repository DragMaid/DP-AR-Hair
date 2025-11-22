# Appearance Feature Extractor

## Overview

The `AppearanceFeatureExtractor` is a neural network encoder that extracts appearance features from input images. It serves dual purposes in the Hair-Shifter pipeline:

1. **Hair Appearance Encoder (E_h)**: Extracts hair-specific visual features
2. **Non-hair Context Encoder (E_c)**: Extracts contextual information from non-hair regions

## Architecture

### Input
- **source_image**: Input image tensor of shape `(B, C, H, W)` where:
  - B = batch size
  - C = image channels (default: 3 for RGB)
  - H, W = spatial dimensions (typically 256x256)

### Output
- **f_s**: 3D feature tensor of shape `(B, reshape_channel, reshape_depth, H', W')` where:
  - reshape_channel = 32 (default)
  - reshape_depth = 16 (default)
  - H', W' = 64x64 (downsampled spatial dimensions)

## Components

### 1. Initial Same Block (`self.first`)
- Applies a standard 2D convolution block without downsampling
- Maps input from 3 channels to `block_expansion` channels
- Kernel size: 3x3 with padding 1
- Output: `(B, block_expansion, H, W)`

### 2. Down Blocks (`self.down_blocks`)
- Series of downsampling blocks reducing spatial dimensions by 2
- Number of blocks: `num_down_blocks`
- Each block:
  - Reduces spatial dimensions by 2 (H×2, W×2 → H, W)
  - Increases channel dimension up to `max_features`
  - Output of block i: `(B, min(max_features, block_expansion * 2^(i+1)), H/(2^(i+1)), W/(2^(i+1)))`

### 3. Second Convolution (`self.second`)
- 1×1 convolution projecting to `max_features` channels
- Prepares features for 3D reshaping
- Output: `(B, max_features, 64, 64)`

### 4. 3D Reshape
- Reshapes 2D feature map into 3D representation
- From: `(B, max_features, 64, 64)`
- To: `(B, reshape_channel, reshape_depth, 64, 64)`
- This creates a pseudo-temporal dimension for 3D processing

### 5. 3D Residual Blocks (`self.resblocks_3d`)
- Series of 3D residual blocks
- Number of blocks: `num_resblocks`
- Applies temporal-spatial convolutions
- Refines 3D feature representation
- Kernel size: 3x3x3 with padding 1

## Parameters

```python
AppearanceFeatureExtractor(
    image_channel=3,           # Input image channels
    block_expansion=64,        # Base channel expansion factor
    num_down_blocks=2,         # Number of downsampling blocks
    max_features=512,          # Maximum feature channels
    reshape_channel=32,        # Channels after 3D reshape
    reshape_depth=16,          # Depth dimension after reshape
    num_resblocks=6            # Number of 3D residual blocks
)
```

## Forward Pass

```
source_image (B, 3, 256, 256)
    ↓
first block
    ↓
down_blocks (spatial reduction)
    ↓
second conv (channel projection)
    ↓ (B, 512, 64, 64)
reshape to 3D
    ↓ (B, 32, 16, 64, 64)
3D residual blocks
    ↓
output f_s (B, 32, 16, 64, 64)
```

## Use Cases

### Hair Appearance Encoder
- Extracts distinctive hair texture and color features
- Used to maintain hair appearance during synthesis
- Condition for hair generation networks

### Non-hair Context Encoder
- Captures facial features and background context
- Preserves identity and structural information
- Used as conditioning for non-hair regions

## Key Characteristics

- **Progressive Downsampling**: Gradually reduces spatial resolution while increasing feature dimensionality
- **3D Feature Representation**: Enables temporal/depth-aware processing through reshape
- **Residual Processing**: 3D residual blocks maintain feature quality through multiple passes
- **Compact Representation**: Reduces computational complexity for downstream tasks

## Usage Example

```python
from models.appearance_feature_extractor import AppearanceFeatureExtractor
import torch

# Initialize extractor
extractor = AppearanceFeatureExtractor(
    image_channel=3,
    block_expansion=64,
    num_down_blocks=2,
    max_features=512,
    reshape_channel=32,
    reshape_depth=16,
    num_resblocks=6
)

# Extract features
source_image = torch.randn(4, 3, 256, 256)  # Batch of 4 images
features = extractor(source_image)
print(features.shape)  # Output: torch.Size([4, 32, 16, 64, 64])
```

## Related Components

- **DownBlock2d**: Implements spatial downsampling with convolution
- **SameBlock2d**: Implements same-resolution convolution
- **ResBlock3d**: Implements 3D residual blocks for feature refinement

## Performance Considerations

- Computational cost scales with:
  - Number of down blocks (spatial reduction)
  - Feature channel dimensions
  - Number of 3D residual blocks
  
- Memory usage increases with batch size and feature depth
- Suitable for GPU acceleration due to large tensor operations

