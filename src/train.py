import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from data.dataset import CelebVHQDataset
from pipelines.training_pipeline import TrainingPipeline
import torch.distributed as dist
from configs.pipeline_config import pipeline_config as pco


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_dir", required=True,
                   help="Folder to find reference hair styles",
                   default=pco.dataset.reference_dir)
    p.add_argument("--drive_dir", required=True,
                   help="Folder to find varied pose faces",
                   default=pco.dataset.driving_dir)
    p.add_argument("--batch_size", type=int,
                   default=pco.training.batch_size)
    p.add_argument("--epochs", type=int,
                   default=pco.training.epoch_num)
    p.add_argument("--num_workers", type=int,
                   default=pco.dataset.num_workers)
    p.add_argument("--save_dir", type=str,
                   default=pco.training.save_dir)
    p.add_argument("--save_every", type=int, help="save every N epochs",
                   default=pco.training.epochs_till_save)
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to resume")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--mixed_precision", action="store_true")
    return p.parse_args()


def main():
    args = get_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    # device
    if args.device:
        device = torch.device(args.device)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    dataset = CelebVHQDataset(driving_dir=args.driving_dir,
                              reference_dir=args.reference_dir,
                              transform=transform,
                              preload=False)

    sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True, drop_last=True, sampler=sampler)

    pipeline = TrainingPipeline(local_rank=local_rank)

    scaler = torch.cuda.amp.GradScaler(
    ) if args.mixed_precision and device.type == "cuda" else None

    start_epoch = 0
    if args.resume:
        ck = pipeline.load_checkpoint(args.resume, load_optimizers=True)
        if "epoch" in ck:
            start_epoch = ck["epoch"] + 1
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        sampler.set(epoch)
        epoch_iterator = tqdm(enumerate(dataloader), total=len(dataloader),
                              desc=f"Epoch {epoch+1}/{args.epochs}")
        running = {"total_loss": 0.0, "disc_loss": 0.0, "steps": 0}
        for step, batch in epoch_iterator:
            # dataset returns (I_s, I_d, I_r)
            I_s, I_d, I_r = batch
            logs = pipeline.train_step(I_s, I_d, I_r, scaler=scaler)

            if dist.get_rank() != 0:
                continue

            running["total_loss"] += logs.get("total_loss", 0.0)
            running["disc_loss"] += logs.get("disc_loss", 0.0)
            running["steps"] += 1

            avg_loss = running["total_loss"] / running["steps"]
            avg_disc = running["disc_loss"] / running["steps"]
            epoch_iterator.set_postfix(
                {"avg_loss": f"{avg_loss:.4f}", "avg_disc": f"{avg_disc:.4f}"})

        # epoch end — checkpoint
        if ((epoch + 1) % args.save_every) == 0:
            if dist.get_rank() == 0:
                ck_path = os.path.join(args.save_dir, f"epoch_{epoch+1:04d}.pt")
                pipeline.save_checkpoint(ck_path, epoch=epoch)
                print(f"Saved checkpoint: {ck_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
