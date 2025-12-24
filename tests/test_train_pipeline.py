import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from pipelines.training_pipeline import TrainingPipeline
from torchvision import transforms as T
from PIL import Image
import pytest

pytestmark = pytest.mark.benchmark_only

# import torch.distributed as dist


@pytest.mark.report_uss
@pytest.mark.report_tracemalloc
@pytest.mark.report_duration
def test_pipeline():
    mp.spawn(ddp_worker, args=(1,), nprocs=1)


def ddp_worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("gloo")
    run_pipeline()
    dist.destroy_process_group()


def run_pipeline():
    device = torch.device("cpu")

    # Minimal transform (resize + to tensor)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    # Initialize pipeline (bypass local_rank / DDP for test)
    class TestPipeline(TrainingPipeline):
        def __init__(self, device):
            self.device = torch.device(device)
            super().__init__(device, loaded=False, generate_on_go=False)

    pipeline = TestPipeline(device=device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    from pathlib import Path
    ck_dir = "./checkpoints_test"
    ck_path = os.path.join(ck_dir, "epoch_0001.pt")
    if Path(ck_path).exists():
        pipeline.load_checkpoint(ck_path, load_optimizers=True)
        print(f"Loaded last checkpoint from {ck_path}")

    # Load 3 images: source, driving, reference
    img_paths = [
        "./assets/test_images/cropped.png",
        "./assets/test_images/cropped.png",
        "./assets/test_images/output.png"
    ]
    images = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0).to(device)  # add batch dimension
        images.append(img)

    I_s, I_d, I_r = images

    # Run a single train step
    with torch.autograd.set_detect_anomaly(True):
        logs = pipeline.train_step(I_s, I_d, I_r, scaler=scaler,
                                   save_debug=True, save_path=Path("./assets/debug_images/"))
    print("Train step logs:", logs)

    # Save minimal checkpoint
    os.makedirs(ck_dir, exist_ok=True)
    pipeline.save_checkpoint(ck_path, epoch=1)
    print(f"Checkpoint saved to {ck_path}")
