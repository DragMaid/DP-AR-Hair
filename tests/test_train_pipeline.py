import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from pipelines.training_pipeline import TrainingPipeline
from torchvision import transforms as T
from PIL import Image
import pytest


@pytest.mark.benchmark
@pytest.mark.report_uss
@pytest.mark.report_tracemalloc
@pytest.mark.report_duration
def test_parallel_pipeline():
    world_size = 2
    mp.spawn(ddp_worker, args=(world_size, 1), nprocs=world_size)


@pytest.mark.benchmark
@pytest.mark.report_uss
@pytest.mark.report_tracemalloc
@pytest.mark.report_duration
@pytest.mark.parametrize("batch_size", [2])
def test_batched_pipeline(batch_size):
    world_size = 1
    mp.spawn(ddp_worker, args=(world_size, batch_size), nprocs=world_size)


def ddp_worker(rank, world_size, batch_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    run_pipeline(device, real_sample=True, batch_size=batch_size)
    dist.destroy_process_group()


def run_pipeline(device, real_sample=False, batch_size=1):
    # Minimal transform (resize + to tensor)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    # Initialize pipeline (bypass local_rank / DDP for test)
    class TestPipeline(TrainingPipeline):
        def __init__(self, device):
            self.device = torch.device(device)
            super().__init__(self.device, loaded=False)

    pipeline = TestPipeline(device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    from pathlib import Path
    ck_dir = "./checkpoints/"
    ck_path = os.path.join(ck_dir, "test.pt")
    if Path(ck_path).exists():
        pipeline.load_checkpoint(ck_path, load_optimizers=True)
        print(f"Loaded last checkpoint from {ck_path}")

    # Load 3 images: source, driving, reference
    images = []
    if real_sample:
        img_paths = [
            "./assets/test_images/cropped.png",
            "./assets/test_images/cropped.png",
            "./assets/test_images/output.png"
        ]
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            img = transform(img)
            img = img.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(
                device)  # add batch dimension
            images.append(img)
    else:
        for _ in range(3):
            images.append(torch.randn([batch_size, 3, 256, 256]))

    I_s, I_d, I_r = images

    with torch.autograd.set_detect_anomaly(True):
        logs = pipeline.train_step(I_s, I_d, I_r, scaler,
                                   mini_batch_size=1,
                                   save_debug=True,
                                   save_path=Path("./assets/debug_images/"))
    print("Train step logs:", logs)

    # Save minimal checkpoint
    os.makedirs(ck_dir, exist_ok=True)
    pipeline.save_checkpoint(ck_path, epoch=1)
    print(f"Checkpoint saved to {ck_path}")
