# Local Losses (HairLoss, ContextLoss)
Location: `src/losses/local_loss.py`

## Overview
This module implements two small, focused L1-based losses that are applied
locally by masks: `HairLoss` and `ContextLoss`.


## Components
- class HairLoss(nn.Module)
- class ContextLoss(nn.Module)

## Description
Both classes are thin wrappers around a masked L1 (L1-norm) difference between
predicted and target images. The inputs follow the convention used throughout
this repository:
- m_h / m_c: mask tensors for hair and context (same spatial size as images)
- I_p: predicted image tensor
- I_d: desired/target image tensor

Each loss computes: torch.norm(mask * (I_d - I_p), p=1)

## Example
```python
import torch
from src.losses.local_loss import HairLoss, ContextLoss

loss_fn = HairLoss()
mask = torch.ones(1, 1, 256, 256)
pred = torch.randn(1, 3, 256, 256)
target = torch.randn(1, 3, 256, 256)
loss = loss_fn(mask, pred, target)
```

## Notes
- Both losses return the L1 norm over masked pixel differences. If your mask
  has multiple channels, broadcasting rules apply.
- These modules are intentionally small and work as drop-in components in
  training loops.
