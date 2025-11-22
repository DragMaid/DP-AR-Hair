# SPADE: Spatially-Adaptive Normalization

## Overview

SPADE (Spatially-Adaptive Normalization) is a normalization technique designed for semantic image synthesis. It normalizes the activations of a network using semantic layout information, enabling better control over the generated image structure. This implementation includes both the `SPADE` normalization layer and `SPADEResnetBlock` for use in generative models.

## Mathematical Background

### SPADE Normalization

SPADE adapts the normalization process by conditioning it on semantic segmentation maps:

```
y = gamma(s) * (x - mean(x)) / sqrt(var(x) + eps) + beta(s)
```

Where:
- `x`: Input feature activations
- `s`: Semantic segmentation map
- `gamma(s)`, `beta(s)`: Scale and bias generated from segmentation
- `mean`, `var`: Computed per-channel, globally across spatial dimensions

## Components

### 1. SPADE Module

```python
SPADE(norm_nc, label_nc)
```

#### Parameters
- **norm_nc**: Number of channels in the feature to be normalized
- **label_nc**: Number of semantic segmentation channels

#### Architecture

```
Segmentation Map (B, label_nc, H, W)
    ↓
MLP Shared Layers
    ├─→ MLP Gamma ──→ Scale (gamma)
    └─→ MLP Beta  ──→ Shift (beta)

Feature Map (B, norm_nc, H, W)
    ↓
InstanceNorm2d (parameter-free)
    ↓
Element-wise Affine Transform using gamma and beta
    ↓
Normalized & Scaled Output
```

#### Key Components

**Parameter-Free Normalization:**
```python
self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
```
- Applies instance normalization without learnable parameters
- Normalizes each sample independently across channels
- Provides stable feature statistics

**Shared MLP:**
```python
self.mlp_shared = nn.Sequential(
    nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
    nn.ReLU()
)
```
- Processes segmentation map with hidden dimension 128
- Creates shared feature representation
- Uses 3×3 convolutions to preserve spatial information

**Gamma and Beta Heads:**
```python
self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
```
- Generate spatially-adaptive scale and shift parameters
- One head per channel, applied element-wise

#### Forward Pass

```python
def forward(self, x, segmap):
    # 1. Normalize input features
    normalized = self.param_free_norm(x)
    
    # 2. Interpolate segmentation to feature size
    segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
    
    # 3. Extract scaling and shift from segmentation
    actv = self.mlp_shared(segmap)
    gamma = self.mlp_gamma(actv)
    beta = self.mlp_beta(actv)
    
    # 4. Apply spatially-adaptive affine transform
    out = normalized * (1 + gamma) + beta
    return out
```

### 2. SPADEResnetBlock

```python
SPADEResnetBlock(fin, fout, norm_G, label_nc, use_se=False, dilation=1)
```

#### Parameters
- **fin**: Number of input channels
- **fout**: Number of output channels
- **norm_G**: Normalization type string (e.g., 'spadespectralinstance')
- **label_nc**: Number of semantic segmentation channels
- **use_se**: Whether to use Squeeze-Excitation (unused in current implementation)
- **dilation**: Dilation factor for convolutions (default: 1)

#### Architecture

```
Input x (B, fin, H, W) + Segmentation seg1 (B, label_nc, H, W)
    ↓
Normalization (SPADE)
    ↓
Activation (LeakyReLU)
    ↓
Conv2d (3×3) with dilation
    ↓
Normalization (SPADE)
    ↓
Activation (LeakyReLU)
    ↓
Conv2d (3×3) with dilation
    ↓
Add Shortcut Connection
    ↓
Output (B, fout, H, W)
```

#### Key Features

**Spectral Normalization:**
- Applied to convolution layers if 'spectral' is in norm_G
- Stabilizes training by constraining Lipschitz constant
- Improves convergence of generative models

**Learned Shortcut:**
```python
self.learned_shortcut = (fin != fout)
if self.learned_shortcut:
    self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
```
- 1×1 convolution for channel adaptation
- Enables residual connections across different channel dimensions
- Preserves gradient flow during backpropagation

**Dilated Convolutions:**
- Supports variable dilation factors
- Enables larger receptive fields without increasing parameters
- Useful for multi-scale feature extraction

#### Forward Pass

```python
def forward(self, x, seg1):
    # Compute shortcut connection
    x_s = self.shortcut(x, seg1)
    
    # Main path with two convolution blocks
    dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))      # First block
    dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))     # Second block
    
    # Residual connection
    out = x_s + dx
    return out
```

## Usage

### Basic SPADE Normalization

```python
from models.spade import SPADE
import torch

# Initialize SPADE normalization
spade = SPADE(norm_nc=256, label_nc=19)

# Input feature and segmentation map
feature = torch.randn(4, 256, 64, 64)  # (B, C, H, W)
segmap = torch.randn(4, 19, 32, 32)    # (B, label_nc, H', W')

# Apply SPADE normalization
normalized = spade(feature, segmap)
print(normalized.shape)  # (4, 256, 64, 64)
```

### SPADE Residual Block

```python
from models.spade import SPADEResnetBlock
import torch

# Initialize SPADE residual block
block = SPADEResnetBlock(
    fin=256,
    fout=256,
    norm_G='spadespectralinstance',
    label_nc=19,
    dilation=1
)

# Forward pass
feature = torch.randn(4, 256, 64, 64)
segmap = torch.randn(4, 19, 64, 64)
output = block(feature, segmap)
print(output.shape)  # (4, 256, 64, 64)
```

### Building a Generator with SPADE Blocks

```python
from models.spade import SPADEResnetBlock
import torch
from torch import nn

class SPADEGenerator(nn.Module):
    def __init__(self, num_blocks=6, label_nc=19):
        super().__init__()
        self.blocks = nn.ModuleList([
            SPADEResnetBlock(256, 256, 'spadespectralinstance', label_nc)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x, segmap):
        for block in self.blocks:
            x = block(x, segmap)
        return x

# Usage
generator = SPADEGenerator()
feature = torch.randn(4, 256, 64, 64)
segmap = torch.randn(4, 19, 64, 64)
output = generator(feature, segmap)
```

## Normalization String Format

The `norm_G` parameter controls normalization behavior:

- **'spadespectralinstance'**: SPADE with spectral normalization on convolutions
- **'spade'**: SPADE without spectral normalization

The string is parsed to determine which techniques to apply to the convolution layers.

## Advantages

1. **Spatial Awareness**: Normalization parameters vary across space based on semantics
2. **Semantic Control**: Segmentation maps directly influence feature generation
3. **Improved Stability**: Spectral normalization prevents training instability
4. **Efficient**: Minimal computational overhead compared to full re-normalization
5. **Flexible**: Works with any semantic segmentation map

## Applications

- **Image-to-Image Translation**: Style transfer conditioned on layout
- **Semantic Image Synthesis**: Generate images from semantic maps
- **Face Generation**: Control facial regions with segmentation
- **Hair Synthesis**: Guide hair generation with segmentation masks
- **Scene Synthesis**: Generate complex scenes from semantic layouts

## Activation Function

```python
def actvn(self, x):
    return F.leaky_relu(x, 2e-1)
```

- Uses Leaky ReLU with slope 0.2
- Allows small gradient flow in negative regions
- Improves training stability

## Related Concepts

- **Batch Normalization**: Channel-wise normalization without spatial adaptation
- **Instance Normalization**: Per-sample channel normalization
- **Group Normalization**: Channel-group normalization
- **Adaptive Instance Normalization (AdaIN)**: Feature-level adaptation
- **Conditional Normalization**: General concept of conditioning on external information

## References

- Original SPADE Paper: "Semantic Image Synthesis with Spatially-Adaptive Normalization"
- Spectral Normalization: "Spectral Normalization for Generative Adversarial Networks"

## Performance Notes

- SPADE adds minimal computational overhead
- Spectral normalization slightly increases memory usage
- Particularly effective for high-resolution image generation
- Works best with multi-class semantic segmentation maps (19-150+ classes)

