# Training launcher
Location: `src/train.py`

## Overview
This script serves as the main training launcher for distributed training using PyTorch's Distributed Data Parallel (DDP) framework. It initializes the distributed environment, sets up mixed precision training if specified, and manages checkpointing and data loading for efficient training across multiple GPUs.

## Data loader usage
- `sampler=DistributedSampler(dataset)`
- `drop_last=True`
- `pin_memory=True`
- `num_workers` set from args
- `batch_size` set from args

## Arguments
- `--mixed_precision` — if set, enables mixed precision (AMP) training when CUDA is available.
- `--resume` — path to a checkpoint to resume from.
- `--save_image_every` — save a debug image every N steps.
- `--save_weight_every` — save a checkpoint every N epochs.
- `--save_dir` — where to write checkpoints.
- `--num_workers` — DataLoader worker processes.
- `--epochs` — number of training epochs.
- `--mini_batch_size` — sub-batch size used inside a single optimizer step (useful for memory-limited GPUs).
- `--batch_size` — global batch size passed to DataLoader.
- `--dataset` — folder containing generated images (used for training stage).
## Read also
- `docs/training_pipeline.md` — detailed explanation of the `TrainingPipeline` class and how it uses DDP, autocast, scalers, and checkpointing.
