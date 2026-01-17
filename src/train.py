import os
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from data.celebvhq_generated import CelebVHQGeneratedDataset
from pipelines.training_pipeline import TrainingPipeline
import torch.distributed as dist
from configs.pipeline_config import pipeline_config as pco


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str,
                   help="Folder to find varied pose faces",
                   default=pco.dataset.datasetdir)
    p.add_argument("--batch_size", type=int,
                   default=pco.training.batch_size)
    p.add_argument("--mini_batch_size", type=int,
                   default=pco.training.mini_batch_size)
    p.add_argument("--epochs", type=int,
                   default=pco.training.epoch_num)
    p.add_argument("--num_workers", type=int,
                   default=pco.dataset.num_workers)
    p.add_argument("--save_dir", type=str,
                   default=pco.training.save_dir)
    p.add_argument("--save_weight_every", type=int, help="save weight every N epochs",
                   default=pco.training.epochs_till_save)
    p.add_argument("--save_image_every", type=int, help="save debug image every N steps",
                   default=pco.training.steps_till_save)
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to resume")
    # p.add_argument("--device", type=str, default=None)
    p.add_argument("--mixed_precision", action="store_true")
    return p.parse_args()


def main():
    args = get_args()

    # TODO: allow user to select a specific device
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    dataset = CelebVHQGeneratedDataset(dataset_dir=args.dataset_dir,
                                       transform=transform)

    sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True, drop_last=True, sampler=sampler)

    # TODO: implement generate on go later, for now its too inefficient
    pipeline = TrainingPipeline(device, local_rank, generate_on_go=False)

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.mixed_precision and device.type == "cuda")

    start_epoch = 0
    if args.resume:
        ck = pipeline.load_checkpoint(args.resume, load_optimizers=True)
        if "epoch" in ck:
            start_epoch = ck["epoch"] + 1
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    # TODO: check if the input retrieval actually works or not
    # TODO: check if epochs is saved automatically
    # TODO: check the unbalanced weight impact
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        epoch_iterator = tqdm(enumerate(dataloader), total=len(dataloader),
                              desc=f"Epoch {epoch+1}/{args.epochs}")
        running = {"total_loss": 0.0, "disc_loss": 0.0, "steps": 0}
        for step, batch in epoch_iterator:
            save_image = (running["steps"]+1) % args.save_image_every == 0
            I_s = batch["front"]["content"]
            I_d = batch["side"]["content"]
            I_r = batch["reference"]["content"]
            logs = pipeline.train_step(
                I_s, I_d, I_r,
                mini_batch_size=args.mini_batch_size,
                scaler=scaler,
                save_debug=save_image,
                save_path=Path("./assets/debug_images/"))

            if dist.get_rank() != 0:
                continue

            running["total_loss"] += logs.get("total_loss", 0.0)
            running["disc_loss"] += logs.get("disc_loss", 0.0)
            running["steps"] += 1

            avg_loss = running["total_loss"] / running["steps"]
            avg_disc = running["disc_loss"] / running["steps"]
            epoch_iterator.set_postfix(
                {"avg_loss": f"{avg_loss:.4f}", "avg_disc": f"{avg_disc:.4f}"})

            # TODO: add mlfow logging here

        # epoch end — checkpoint
        if ((epoch + 1) % args.save_weight_every) == 0:
            if dist.get_rank() == 0:
                ck_path = os.path.join(
                    args.save_dir, f"epoch_{epoch+1:04d}.pt")
                pipeline.save_checkpoint(ck_path, epoch=epoch)
                print(f"Saved checkpoint: {ck_path}")

    dist.destroy_process_group()
    print("Training complete.")


if __name__ == "__main__":
    main()
