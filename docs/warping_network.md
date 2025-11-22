# Warping Network

## Overview

The `WarpingNetwork` is a neural network module that implements spatial deformation and feature warping capabilities. It serves as the Warping module (W) in the Hair-Shifter paper, enabling the transformation of 3D features from a source space to a driving space through dense motion estimation and occlusion handling.

## Architecture

### Input
- **feature_3d**: 3D feature tensor of shape `(B, C, D, H, W)` where:
  - B = batch size
  - C = feature channels (typically 32)
  - D = depth/temporal dimension (typically 16)
  - H, W = spatial dimensions (typically 64x64)

- **kp_driving**: Driving keypoints of shape `(B, num_kp, 3)`
- **kp_source**: Source keypoints of shape `(B, num_kp, 3)`

### Output
Dictionary containing:
- **'out'**: Warped feature tensor `(B, C, H, W)` - the deformed 3D features projected to 2D
- **'occlusion_map'**: Occlusion mask `(B, 1, H, W)` - indicates visible regions
- **'deformation'**: Deformation field `(B, D, H, W, 3)` - spatial displacement vectors

## Components

### 1. Dense Motion Network

```python
self.dense_motion_network = DenseMotionNetwork(
    num_kp=num_kp,
    feature_channel=reshape_channel,
    estimate_occlusion_map=estimate_occlusion_map,
    **dense_motion_params
)
```

The Dense Motion Network computes:
- **Sparse Motions**: Motion vectors for each keypoint
- **Deformation Field**: Dense pixel-wise deformation by interpolating keypoint motions
- **Occlusion Map**: Confidence map indicating which regions are visible

#### Dense Motion Computation Process

```
1. Sparse Motion Creation
   - Keypoint matching: driving_kp → source_kp
   - For each pixel, compute motion from nearest keypoints
   
2. Heatmap Representation
   - Gaussian heatmaps for each keypoint
   - Difference between driving and source heatmaps
   
3. Hourglass Network
   - Multi-scale feature processing
   - Refinement of motion estimates
   
4. Mask Generation
   - Softmax normalization across keypoints
   - Weighted motion field
   
5. Occlusion Estimation
   - Per-pixel occlusion confidence
   - Handles self-occlusions and background
```

### 2. Feature Reshaping Layers

```python
self.third = SameBlock2d(max_features, 
                         block_expansion * (2 ** num_down_blocks),
                         kernel_size=(3, 3), 
                         padding=(1, 1), 
                         lrelu=True)

self.fourth = nn.Conv2d(in_channels=in_channels_dim,
                        out_channels=out_channels_dim,
                        kernel_size=1,
                        stride=1)
```

These layers process the warped 3D features:
- **third block**: Standard 2D convolution with LeakyReLU activation
- **fourth layer**: Channel-wise projection (1×1 convolution)

## Forward Pass

```
Input: feature_3d (B, C, D, H, W), kp_driving, kp_source
    ↓
Dense Motion Network
    ├─→ Compute sparse motions
    ├─→ Create heatmap representations
    ├─→ Deform features via grid sampling
    ├─→ Generate occlusion map (optional)
    └─→ Output deformation field
    
Deformation = dense_motion['deformation']  # (B, D, H, W, 3)
Occlusion = dense_motion['occlusion_map']  # (B, 1, H, W)
    ↓
Grid Sample: Warp feature_3d using deformation
    → Warped Features: (B, C, D, H, W)
    ↓
Reshape: Collapse depth dimension
    → (B, C*D, H, W) = (B, 512, 64, 64)
    ↓
Third Block (SameBlock2d)
    → (B, 256, 64, 64)
    ↓
Fourth Layer (1×1 Conv)
    → (B, 256, 64, 64)
    ↓
Optional Occlusion Masking
    → out * occlusion_map if flag_use_occlusion_map
    ↓
Output Dictionary:
    - 'out': (B, 256, 64, 64)
    - 'deformation': (B, D, H, W, 3)
    - 'occlusion_map': (B, 1, H, W)
```

## Parameters

```python
WarpingNetwork(
    num_kp=15,                          # Number of keypoints
    block_expansion=64,                 # Base channel expansion
    max_features=256,                   # Maximum feature channels
    num_down_blocks=2,                  # Downsampling blocks in encoder
    reshape_channel=32,                 # Channels after 3D reshape
    estimate_occlusion_map=False,       # Whether to estimate occlusion
    dense_motion_params={               # Parameters for DenseMotionNetwork
        'block_expansion': 64,
        'num_blocks': 4,
        'max_features': 256,
        'compress': 4,
        'reshape_depth': 16,
    },
    upscale=1,                          # Upscaling factor
    flag_use_occlusion_map=True,        # Whether to use occlusion in output
)
```

## Static Methods

### `deform_input(inp, deformation)`

Applies spatial deformation to input features using grid sampling.

```python
@staticmethod
def deform_input(inp, deformation):
    return F.grid_sample(inp, deformation, align_corners=False)
```

**Parameters:**
- `inp`: Input feature tensor `(B, C, D, H, W)`
- `deformation`: Deformation field `(B, D, H, W, 3)` with normalized coordinates in [-1, 1]

**Returns:**
- Warped features using bilinear interpolation

**Key Points:**
- Uses bilinear interpolation for smooth warping
- Requires deformation coordinates in [-1, 1] range
- `align_corners=False` uses PyTorch's standard grid sampling

## Key Mechanisms

### 1. Spatial Deformation
- Motion vectors guide pixel displacement
- Grid sampling performs interpolation
- Enables continuous spatial transformation

### 2. Occlusion Handling
```python
if self.flag_use_occlusion_map and (occlusion_map is not None):
    out = out * occlusion_map
```
- Masks warped regions with confidence values
- Reduces artifacts in occluded areas
- Can be disabled if occlusion estimation is unavailable

### 3. Dimension Collapsing
```python
bs, c, d, h, w = out.shape  # (B, C, D, H, W)
out = out.view(bs, c * d, h, w)  # (B, C*D, H, W)
```
- Converts 3D features to 2D representation
- Preserves depth information in channel dimension
- Enables 2D processing of 3D features

## Use Cases

### Motion-Driven Feature Transformation
- Adapt appearance features based on head motion
- Warp hairstyle features to match new pose
- Enable realistic motion in generated videos

### Face Reenactment
- Transfer facial expressions via keypoint matching
- Maintain appearance consistency across frames
- Handle occlusions gracefully

### Hair Transfer
- Guide hair generation based on head movement
- Preserve hair structure during pose changes
- Maintain visual quality under transformation

## Advantages

1. **Keypoint-Based Control**: Explicit motion via keypoints
2. **Occlusion Awareness**: Handles self-occlusions
3. **Dense Coverage**: Per-pixel deformation through interpolation
4. **End-to-End Differentiable**: Enables gradient-based optimization
5. **Multi-scale Processing**: Dense motion network uses hourglass for multi-scale features

## Output Interpretation

### Warped Features (`out`)
- Transformed appearance features in driving pose
- Used as input for downstream synthesis modules
- Shape: `(B, C, H, W)` typically `(B, 256, 64, 64)`

### Deformation Field
- Spatial displacement vectors for each pixel
- Range: [-1, 1] normalized coordinates
- Shape: `(B, D, H, W, 3)` for 3D deformation

### Occlusion Map
- Confidence/visibility scores
- Range: [0, 1] after sigmoid
- Shape: `(B, 1, H, W)`
- Values near 1 indicate visible regions
- Values near 0 indicate occluded regions

## Integration with Other Modules

```
Input Image
    ↓
AppearanceFeatureExtractor → feature_3d
MotionExtractor → keypoints
    ↓
WarpingNetwork(feature_3d, kp_driving, kp_source)
    ↓
warped_features → ContextDecoder
    ↓
Final Output Image
```

## Performance Considerations

- **Computational Cost**:
  - Dense motion network: ~60G (primary cost)
  - Grid sampling: ~0.5G
  - Feature projection: ~1G
  
- **Memory Usage**:
  - Grows with batch size and spatial resolution
  - 3D feature storage is significant
  
- **Optimization**:
  - Can reduce num_kp for efficiency
  - Upscale parameter enables resolution control
  - Can disable occlusion estimation if not needed

## Related Components

- **DenseMotionNetwork**: Core motion estimation module
- **SameBlock2d**: Standard 2D convolution block
- **grid_sample**: PyTorch's differentiable spatial sampling
- **make_coordinate_grid**: Creates reference coordinate grids
- **kp2gaussian**: Converts keypoints to Gaussian heatmaps

## Notes

- Deformation field assumes normalized coordinates [-1, 1]
- Grid sampling is differentiable, enabling end-to-end training
- Occlusion map improves realism but adds computational cost
- Works best with well-defined keypoint sets

