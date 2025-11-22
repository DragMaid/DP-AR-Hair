# Motion Extractor

## Overview

The `MotionExtractor` is a neural network encoder that extracts motion and pose features from images. It serves as the motion feature encoder (E_m) in the Hair-Shifter pipeline, detecting and encoding facial motion, head pose, and expression information.

## Architecture

The Motion Extractor is built upon **ConvNeXtV2 Tiny**, a state-of-the-art vision transformer-based backbone that provides robust feature extraction with excellent efficiency.

### Input
- **x**: Input image tensor of shape `(B, C, H, W)` where:
  - B = batch size
  - C = 3 (RGB channels)
  - H, W = spatial dimensions (typically 256x256)

### Output
- **out**: Dictionary containing multiple motion-related predictions:
  - **pitch**: Head pitch angle (angle bins, shape: `(B, num_bins)`)
  - **yaw**: Head yaw angle (angle bins, shape: `(B, num_bins)`)
  - **roll**: Head roll angle (angle bins, shape: `(B, num_bins)`)
  - **t**: Translation vector (shape: `(B, 3)`)
  - **exp**: Expression/delta keypoints (shape: `(B, 3*num_kp)`)
  - **scale**: Scale factor (shape: `(B, 1)`)
  - **kp**: Implicit canonical keypoints (shape: `(B, 3*num_kp)`)

## Component Breakdown

### ConvNeXtV2 Tiny Backbone

The detector uses ConvNeXtV2 Tiny with the following architecture:

#### Stage Configuration
```
Stage 0: 3 blocks, 96 channels  (stride 4 downsampling)
Stage 1: 3 blocks, 192 channels (stride 2 downsampling)
Stage 2: 9 blocks, 384 channels (stride 2 downsampling)
Stage 3: 3 blocks, 768 channels (stride 2 downsampling)
```

#### Stem
- Input convolution: 3 → 96 channels
- Kernel: 4x4, stride 4
- Layer normalization

#### Feature Extraction
- ConvNeXtV2 blocks with:
  - 7×7 depthwise convolutions
  - Pointwise linear projections
  - Global Response Normalization (GRN)
  - GELU activations
  - Residual connections
  - Optional stochastic depth regularization

#### Output Heads
After global average pooling and layer normalization:

1. **Keypoint Head** (`fc_kp`):
   - Linear layer: 768 → 3*num_kp
   - Implicit keypoint coordinates
   - Default num_kp: 24 keypoints

2. **Pose Heads**:
   - **Pitch** (`fc_pitch`): 768 → num_bins
   - **Yaw** (`fc_yaw`): 768 → num_bins
   - **Roll** (`fc_roll`): 768 → num_bins
   - Default num_bins: 66 angle bins

3. **Translation Head** (`fc_t`):
   - Linear layer: 768 → 3
   - 3D translation vector

4. **Expression Head** (`fc_exp`):
   - Linear layer: 768 → 3*num_kp
   - Expression deformation/delta keypoints

5. **Scale Head** (`fc_scale`):
   - Linear layer: 768 → 1
   - Scale normalization factor

## Parameters

```python
MotionExtractor(
    num_kp=24,       # Number of implicit keypoints
    num_bins=66      # Number of angle bins for pose
)
```

## Forward Pass

```
Input Image (B, 3, H, W)
    ↓
Stem: Conv2d (stride 4) + LayerNorm
    ↓ (B, 96, H/4, W/4)
Stage 0: 3 ConvNeXtV2 blocks
    ↓ (B, 96, H/4, W/4)
DownSample + Stage 1: 3 blocks
    ↓ (B, 192, H/8, W/8)
DownSample + Stage 2: 9 blocks
    ↓ (B, 384, H/16, W/16)
DownSample + Stage 3: 3 blocks
    ↓ (B, 768, H/32, W/32)
Global Average Pooling
    ↓ (B, 768)
LayerNorm
    ↓
Parallel Head Projections
    ↓
Output Dictionary
```

## Methods

### `load_pretrained(init_path: str)`

Loads pretrained weights from a checkpoint file.

**Parameters:**
- `init_path`: Path to pretrained model checkpoint
  - Expected checkpoint format: Contains a 'model' key with state dict
  - The 'head' layer is automatically filtered out
  - Uses non-strict loading to allow architecture differences

**Example:**
```python
extractor = MotionExtractor()
extractor.load_pretrained('path/to/checkpoint.pth')
```

### `forward(x)`

Extracts motion features from input image.

**Parameters:**
- `x`: Input image tensor `(B, C, H, W)`

**Returns:**
- Dictionary with keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'

## Use Cases

### Motion Feature Extraction
- **Head Pose Detection**: Estimates 3-DOF head rotation (pitch, yaw, roll)
- **Facial Keypoints**: Detects implicit canonical keypoints (24 points)
- **Expression**: Captures facial expressions via expression deformations
- **Translation**: Estimates 3D head position shifts
- **Scale**: Normalizes for face size variations

### Hair Transfer Pipeline
- Provides motion conditioning for hair synthesis
- Enables pose-aware hair generation
- Maintains expression consistency in output

## Key Characteristics

- **Modern Architecture**: Uses ConvNeXtV2, a recent state-of-the-art vision model
- **Multi-task Learning**: Simultaneously predicts pose, expression, keypoints, and scale
- **Efficient**: Tiny variant optimized for real-time inference
- **Pretrained Support**: Can load ImageNet or task-specific pretrained weights
- **Rich Output**: Multiple complementary representations of motion

## Usage Example

```python
from models.motion_extractor import MotionExtractor
import torch

# Initialize extractor
extractor = MotionExtractor(num_kp=24, num_bins=66)

# Load pretrained weights
extractor.load_pretrained('pretrained/motion_extractor.pth')

# Extract motion features
input_image = torch.randn(4, 3, 256, 256)
motion_dict = extractor(input_image)

print(f"Pitch shape: {motion_dict['pitch'].shape}")     # (4, 66)
print(f"Keypoints shape: {motion_dict['kp'].shape}")    # (4, 72) - 24*3
print(f"Expression shape: {motion_dict['exp'].shape}")  # (4, 72) - 24*3
print(f"Translation shape: {motion_dict['t'].shape}")   # (4, 3)
print(f"Scale shape: {motion_dict['scale'].shape}")     # (4, 1)
```

## Related Components

- **ConvNeXtV2**: Modern vision transformer backbone
- **LayerNorm**: Normalization layer with optional channel-first format
- **DropPath**: Stochastic depth regularization
- **GRN**: Global Response Normalization for better generalization

## Performance Considerations

- **Inference Speed**: Optimized for real-time processing (Tiny variant)
- **Memory Usage**: Moderate, suitable for GPU inference
- **Batch Processing**: Supports variable batch sizes
- **Scalability**: Can use larger ConvNeXtV2 variants (Small, Base) for higher accuracy

## Notes

- The 66 angle bins provide fine-grained head pose estimation
- The 24 implicit keypoints represent both explicit face landmarks and implicit features
- Translation and scale enable full 6-DOF head pose representation
- Expression deltas allow non-rigid facial deformation modeling

