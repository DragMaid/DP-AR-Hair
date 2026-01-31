import torch
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from pipelines.training_pipeline import TrainingPipeline
from configs.pipeline_config import pipeline_config as pco
from utils import MLFlowManager
from losses.utils import StepLogger
from torchvision.utils import make_grid, save_image
from torchvision import transforms as T
from matplotlib import pyplot as plt
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
@pytest.mark.parametrize("batch_size", [1])
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
        def __init__(self, device, logger):
            super().__init__(device, logger, loaded=False)

    logger = StepLogger()
    pipeline = TestPipeline(device=device, logger=logger)
    first_processor = dist.get_rank() == 0

    # Refering to entire pipeline beside IIHT
    generator_optimizer = torch.optim.Adam(
        pipeline.generator_trainable_params,
        lr=pco.training.generator.learn_rate,
        betas=pco.training.generator.betas)

    # Discrimination optmizer
    disc_optimizer = torch.optim.Adam(
        pipeline.L_adv.parameters(),
        lr=pco.training.discriminator.learn_rate,
        betas=pco.training.discriminator.betas)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    pipeline.set_optimizers(
        generator_optimizer=generator_optimizer,
        disc_optimizer=disc_optimizer
    )

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
            "./assets/test_images/side.jpg",
            "./assets/test_images/front.jpg",
            "./assets/test_images/generated.jpg"
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

    I_s, I_d, I_d_dilde = images

    logger.reset()

    mlflow_manager = MLFlowManager()

    with mlflow_manager.start_run():
        with torch.autograd.set_detect_anomaly(True):
            pipeline.train_step(
                I_s, I_d, I_d_dilde,
                scaler,
                mini_batch_size=1,
                accumulate_grad_contrib=True,
                store_outputs=True)

        if first_processor:
            logger.calculate_grad_norms(pipeline.modules_to_log)

            logger.snapshot_params(
                "generator", pipeline.generator_trainable_params)
            logger.snapshot_params(
                "discriminator", pipeline.disc_trainable_params)

        # Update the weights and reset optimizer
        scaler.step(disc_optimizer)
        scaler.step(generator_optimizer)
        disc_optimizer.zero_grad(set_to_none=True)
        generator_optimizer.zero_grad(set_to_none=True)
        scaler.update()

        if not first_processor:
            return

        logger.log_param_update(
            "generator", pipeline.generator_trainable_params)
        logger.log_param_update(
            "discriminator", pipeline.disc_trainable_params)

        # Calculate param norms
        logger.calculate_param_norms(pipeline.modules_to_log)

        # Calculate parameters distribution
        logger.log_param_distribution(
            "generator", pipeline.generator_trainable_params)
        logger.log_param_distribution(
            "discriminator", pipeline.disc_trainable_params)

        logs = logger.finalize()
        output_images = logs.pop("output_images")

        # Plot the contribution bar plot
        grad_contrib_ratios = logs["gradient_contribs"]
        plt.figure()
        plt.title("Loss gradient contribution")
        headers = [k.split('/')[-1][:4]
                   for k in grad_contrib_ratios.keys()]
        plt.bar(headers, grad_contrib_ratios.values())
        file_path = Path("assets/artifacts/contribs/test.jpg")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)

        # Plot the batch output images
        grid = make_grid(output_images, nrow=8, normalize=True)
        file_path = Path("assets/artifacts/outputs/step_test.png")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, file_path)

        print(logs)

        # Save minimal checkpoint
        os.makedirs(ck_dir, exist_ok=True)
        pipeline.save_checkpoint(ck_path, epoch=1)
        print(f"Checkpoint saved to {ck_path}")
