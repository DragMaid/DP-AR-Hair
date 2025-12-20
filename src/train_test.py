import torch
import os
import torch.distributed as dist
from pipelines.training_pipeline import TrainingPipeline
from torchvision import transforms as T
from PIL import Image


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    # Minimal transform (resize + to tensor)
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])

    # Initialize pipeline (bypass local_rank / DDP for test)
    class TestPipeline(TrainingPipeline):
        def __init__(self, device):
            self.device = torch.device(device)
            super().__init__(local_rank=0)  # still needs an int for original init

    pipeline = TestPipeline(device=device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Load 3 images: source, driving, reference
    img_paths = [
        "./assets/test_images/phuc.jpeg",
        "./assets/test_images/phuc.jpeg",
        "./assets/test_images/ken.png"
    ]
    images = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0).to(device)  # add batch dimension
        images.append(img)

    I_s, I_d, I_r = images

    # Run a single train step
    logs = pipeline.train_step(I_s, I_d, I_r, scaler=scaler)
    print("Train step logs:", logs)

    # Save minimal checkpoint
    ck_dir = "./checkpoints_test"
    os.makedirs(ck_dir, exist_ok=True)
    ck_path = os.path.join(ck_dir, "epoch_0001.pt")
    pipeline.save_checkpoint(ck_path, epoch=1)
    print(f"Checkpoint saved to {ck_path}")


if __name__ == "__main__":
    main()
