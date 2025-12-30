# TrainingPipeline
Location: `src/pipelines/training_pipeline.py`

## Overview
`TrainingPipeline` encapsulates the model components, optimizers, loss handling, distributed wrappers, and the logic for a single training step. It is the central place where generator and discriminator models are combined, losses computed, and optimizer steps taken.

## Key responsibilities
- Load and wrap model components (encoders, warping, SPADE decoders, discriminators).
- Wrap trainable modules with `torch.nn.parallel.DistributedDataParallel` (DDP) when running on multiple processes/GPUs.
- Provide `train_step(...)` which performs forward passes, computes losses (generator and discriminator), and performs optimizer steps including gradient scaling for AMP.
- Provide checkpoint save/load utilities (`save_checkpoint` / `load_checkpoint`).

## DistributedDataParallel (DDP)
DDP is used to wrap trainable modules so that gradients are synchronized across processes after the backward pass. Important notes:

- Device assignment: When running on GPUs, `local_rank` is passed to DDP via `device_ids` and `output_device`. The training script sets the CUDA device before constructing the pipeline to ensure each process uses the correct GPU.
- Wrapping pattern: Only trainable modules (those with parameters that require gradients) and modules which must synchronize buffers are wrapped. In this project `E_C` and the combined decoder `D` (MSGSpadeDecoder) are wrapped with DDP.
- Non-trainable modules can remain unwrapped if they do not require gradient synchronization.

## Automatic Mixed Precision (autocast) and GradScaler
This pipeline uses AMP via two mechanisms:
- `torch.cuda.amp.autocast(enabled=...)` is used to run forward passes in mixed precision where beneficial. Casting to lower precision reduces memory usage and can accelerate ops on modern GPUs.
- `torch.cuda.amp.GradScaler` is used to scale the loss before backpropagation to preserve small gradients and avoid underflow when using float16.

- The pipeline optionally expects a `GradScaler` object to be passed in from the outer training script (so a single scaler can be shared across multiple gradient accumulation steps).
- The pipeline uses `autocast` around the parts of the forward pass that are safe to run in mixed precision (model inference and generator forward).

## Mini-batching / gradient accumulation
`train_step` supports splitting the incoming batch into smaller `mini_batch_size` chunks. This enables effectively larger batch sizes than what fits into GPU memory by accumulating gradients across multiple mini-batches within a single optimizer step. Important details:

- The discriminator and generator losses are scaled by `1 / batch_size` before backward so the accumulated gradients reflect an average across the full batch.
- `.step()` and `.zero_grad()` are called only on the final mini-batch for that overall batch, allowing gradient accumulation across iterations.

## Train flow
Typical forward flow in a single mini-batch inside `train_step`:
1. Compute appearance features f_c from the trainable encoder `E_C`.
2. With `torch.no_grad()` compute auxiliary features used for warping: f_h from `E_H`, keypoints from `E_M` for source and driving, and warped features from `W`.
3. Compute a binary mask for the hair class using face parsing helper `get_mask_by_idx`.
4. Run the decoder `D` (MSGSpadeDecoder) to synthesize the predicted image `I_p` from appearance and warped features.
5. Compute discriminator loss on real driving images vs generated prediction using the `L_adv` discriminator.
6. Compute generator losses (perceptual, L1, adversarial, etc.) via `LossHandler`.

## Discriminator and generator updates
- The discriminator is updated using a standard backward/step pattern, but the script divides the discriminator loss by `batch_size` and only calls `.step()` after processing all mini-batches for the combined full batch (to match the gradient scaling approach used for generators).
- Generator losses are likewise averaged and backpropagated using the GradScaler.

## Checkpointing: save_checkpoint and load_checkpoint
- `save_checkpoint(path, epoch, extra=None)` collects state dictionaries from the `modules_to_save` mapping and optimizer states and writes a single payload to disk using `torch.save`.
- Module state loading uses `strict=False` where possible to allow for slight shape/architecture mismatches, falling back to a strict load if needed. Optimizer states are optionally restored when `load_optimizers=True`.

## Debugging and saving images
When `save_debug=True` and the current mini-batch is the final mini-batch of the batch, the pipeline will save a normalized debug image (predicted image scaled from [-1, 1] to [0, 1]) into the provided `save_path`. This helps visually inspect outputs during training.

## Notes
- Use torchrun to launch instances when training instead of writing custom launch code. Example:
  ```bash
  torchrun --nproc_per_node=4 train.py --batch_size 16 --mini_batch_size 4 --mixed_precision
  ```
- The DDP wrap and device set to "cpu" is entirely for testing purposes, using it for training is impractical.
- Use `mini_batch_size` carefully: too small will increase synchronization overhead; too large will use more memory.
- If you see numerical instability with AMP, try disabling AMP to debug, or adjust loss scaling strategy.

## See also
- `docs/train.md` — how `train.py` orchestrates distributed training and AMP.
- The repository `docs/` pages for individual model components (encoders, warper, decoders) for details on expected inputs and shapes.

