# train.py
import argparse
import torch
from torch.utils.data import DataLoader
from data.dataset import CelebVHQDataset
from pipelines.training_pipeline import TrainingPipeline
from torchvision import transforms as T
from tqdm import tqdm
import os


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="path to celebvhq_info.json")
    p.add_argument("--video_root", required=True,
                   help="processed video root directory")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=1,
                   help="save every N epochs")
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to resume")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--mixed_precision", action="store_true")
    return p.parse_args()


def main():
    args = get_args()

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms (applied by dataset if provided)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),   # adjust to your network's expected size
        T.ToTensor(),           # this yields CHW float in [0,1]
    ])

    dataset = CelebVHQDataset(json_path=args.json,
                              processed_video_root=args.video_root,
                              transform=transform,
                              preload=False)

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

    pipeline = TrainingPipeline(device=device)

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
        epoch_iterator = tqdm(enumerate(dataloader), total=len(dataloader),
                              desc=f"Epoch {epoch+1}/{args.epochs}")
        running = {"total_loss": 0.0, "disc_loss": 0.0, "steps": 0}
        for step, batch in epoch_iterator:
            # dataset returns (I_s, I_d, I_r)
            I_s, I_d, I_r = batch
            logs = pipeline.train_step(I_s, I_d, I_r,  scaler=scaler)

            running["total_loss"] += logs.get("total_loss", 0.0)
            running["disc_loss"] += logs.get("disc_loss", 0.0)
            running["steps"] += 1

            avg_loss = running["total_loss"] / running["steps"]
            avg_disc = running["disc_loss"] / running["steps"]
            epoch_iterator.set_postfix(
                {"avg_loss": f"{avg_loss:.4f}", "avg_disc": f"{avg_disc:.4f}"})

        # epoch end — checkpoint
        if ((epoch + 1) % args.save_every) == 0:
            ck_path = os.path.join(args.save_dir, f"epoch_{epoch+1:04d}.pt")
            pipeline.save_checkpoint(ck_path, epoch=epoch)
            print(f"Saved checkpoint: {ck_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
