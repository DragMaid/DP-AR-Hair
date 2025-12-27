# LossHandler — centralized loss computation
Location: `src/losses/loss_handler.py`

## Overview
`LossHandler` encapsulates the loss modules used to train the generator and discriminator. It owns loss submodules (perceptual, local hair/face losses, global L1) and the discriminator criterion, and exposes two main helpers used by training code:

- `compute_generator_losses(...)` — computes perceptual, local, global and adversarial losses and returns a weighted sum plus the individual terms.
- `compute_discriminator_loss(...)` — computes the discriminator loss distinguishing real vs fake images.

This file documents the expected inputs/outputs, key implementation details (AMP, device handling), configuration hooks, and suggestions for testing and debugging.

## Contract
**Inputs** (both functions expect tensors on the same device as the `LossHandler`):
- I_d: real target image tensor, shape (B, C, H, W), normalized in the same range as training (repo usually uses [-1,1] or [0,1] depending on pipeline — ensure consistency with your pipeline).
- I_p / I_p_detached: predicted image tensor from the generator, shape (B, C, H, W). For generator loss it may be used as the differentiable prediction, for discriminator loss it must be detached.
- m_c: hair mask tensor, shape (B, 1, H, W) (used by hair-local loss).
- m_f: face / non-hair mask tensor, shape (B, 1, H, W) (used by face-local loss).
- discriminator: callable module (nn.Module) that produces logits for adversarial losses.

**Outputs**:
- `compute_generator_losses` returns a dict with keys: `total_loss`, `p_loss`, `h_loss`, `f_loss`, `g_loss`, `a_gen_loss`. `total_loss` is a weighted sum using weights defined in the pipeline config.
- `compute_discriminator_loss` returns a scalar tensor containing the discriminator loss averaged over real and fake components.

## Loss modules and weights
- Perceptual: `PerceptualLoss` (VGG-based feature-space loss). Input images are normalized with ImageNet mean/std via torchvision transforms.Normalize stored as `self.normalize`.
- Local losses: `HairLoss` and `FaceLoss` compute local-region-specific penalties.
- Global L1: standard reconstruction loss `nn.L1Loss`.
- Discriminator loss: `nn.BCEWithLogitsLoss` is used on the discriminator logits for stability.

Weights are read from `configs.pipeline_config.pipeline_config.training.loss` and applied as:
- total = p_rate * p_loss + adv_rate * a_gen_loss + h_rate * h_loss + f_rate * f_loss + rec_rate * g_loss

## AMP (autocast) and device handling
- The handler uses `torch.cuda.amp.autocast` when the configured device is CUDA: `with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):`. This accelerates perceptual loss evaluation and can reduce memory.
- The handler stores all loss modules on the provided `device` during construction (e.g., `PerceptualLoss().to(device)`). Ensure the `LossHandler` is constructed with the same device used by the model and that inputs are moved to that device before calling the loss functions.

## Adversarial details
- Generator adversarial loss: the predicted images are forwarded through the discriminator and compared to a `target_real` tensor (ones) via BCEWithLogitsLoss. This encourages the generator to make discriminator outputs close to real.
- Discriminator loss: evaluates discriminator on real images (`target_real`) and fake images (`target_fake` zeros), averages the two losses.

## Important implementation notes
- The perceptual loss call normalizes inputs with ImageNet mean/std using the stored `self.normalize`. If your model uses images scaled to [-1,1] you must ensure the normalization matches VGG preprocessing expectations (i.e., map to [0,1] then normalize). The code assumes the tensors passed in are in a compatible range.
- For discriminator evaluation the code uses `torch.ones_like(pred)` and `torch.zeros_like(pred)` to create targets with the exact same shape.
- Loss modules are stateful PyTorch `nn.Module`s but `LossHandler` treats them as stateless helpers (no parameters are expected to be optimized from these modules, except where the modules themselves have parameters).

## Examples
```python
# Create handler on CPU:
handler = LossHandler(device=torch.device('cpu'))
# Create dummy tensors (B=2, C=3, H=64, W=64):
I_d = torch.rand(2,3,64,64)
I_p = torch.rand_like(I_d)
m_c = torch.randint(0,2,(2,1,64,64)).float()
m_f = 1.0 - m_c
# Dummy discriminator:
class D(nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), 1)
# Call generator loss:
losses = handler.compute_generator_losses(I_d, I_p, m_c, m_f, D())
assert set(losses.keys()) >= {"total_loss","p_loss","h_loss","f_loss","g_loss","a_gen_loss"}
```

See also
---
- `src/losses/perceptual_loss.py` — perceptual / VGG-based feature loss implementation.
- `src/losses/local_loss.py` — hair and face localized loss terms.
- `configs/pipeline_config.py` — loss weight configuration used by `LossHandler`.


