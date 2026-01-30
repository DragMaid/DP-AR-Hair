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
from utils import MLFlowManager
from losses.utils import StepLogger
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt
from datetime import datetime


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str,
                   help="Folder to find varied pose faces",
                   default=pco.dataset.dataset_dir)
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

    # TODO: Add the proper URL instead of the default localhost:5000 (PostgreSQL)
    mlflow_manager = MLFlowManager()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    dataset = CelebVHQGeneratedDataset(dataset_dir=args.dataset,
                                       transform=transform)

    sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True, drop_last=True, sampler=sampler)

    logger = StepLogger()
    pipeline = TrainingPipeline(device, logger, local_rank)

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

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.mixed_precision and device.type == "cuda")

    # Set the optimizers to be used in pipeline
    pipeline.set_optimizers(
        generator_optimizer=generator_optimizer,
        disc_optimizer=disc_optimizer
    )

    start_epoch = 0
    if args.resume:
        ck = pipeline.load_checkpoint(args.resume, load_optimizers=True)
        if "epoch" in ck:
            start_epoch = ck["epoch"] + 1
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)

    with mlflow_manager.start_run():
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            epoch_iterator = tqdm(enumerate(dataloader), total=len(dataloader),
                                  desc=f"Epoch {epoch+1}/{args.epochs}")

            global_step = 0
            for step, batch in epoch_iterator:
                grad_contrib = (
                    global_step+1) % pco.training.log_config.grad_contrib_interval == 0 \
                    and dist.get_rank() == 0
                param_norm = (
                    global_step+1) % pco.training.log_config.param_norm_interval == 0 \
                    and dist.get_rank() == 0
                param_dist = (
                    global_step+1) % pco.training.log_config.param_dist_interval == 0 \
                    and dist.get_rank() == 0
                save_debug = (
                    global_step+1) % pco.training.log_config.output_save_interval == 0 \
                    and dist.get_rank() == 0

                # Get the 3 images: original front / side and hair transfered image
                I_s = batch["reference"]["content"]
                I_d = batch["driving"]["content"]
                I_d_dilde = batch["generated"]["content"]

                if param_dist:
                    logger.snapshot_params(
                        "generator", pipeline.generator_trainable_params)
                    logger.snapshot_params(
                        "discriminator", pipeline.disc_trainable_params)

                pipeline.train_step(
                    I_s, I_d, I_d_dilde,
                    mini_batch_size=args.mini_batch_size,
                    scaler=scaler,
                    accumulate_grad_contrib=grad_contrib)

                # Log gradient norms every step
                if dist.get_rank() == 0:
                    logger.calculate_grad_norms(pipeline.modules_to_log)

                # Update the weights and reset optimizer
                scaler.step(disc_optimizer)
                scaler.step(generator_optimizer)
                disc_optimizer.zero_grad(set_to_none=True)
                generator_optimizer.zero_grad(set_to_none=True)
                scaler.update()

                if param_norm:
                    logger.calculate_param_norms(pipeline.modules_to_log)

                if param_dist:
                    logger.log_param_distribution(
                        "generator", pipeline.generator_trainable_params)
                    logger.log_param_distribution(
                        "discriminator", pipeline.disc_trainable_params)

                if dist.get_rank() == 0:
                    logs = logger.finalize()
                    for scalar_log in ["losses", "gradient_norms", "param_norms"]:
                        for k, v in logs[scalar_log].items():
                            mlflow_manager.log_metric(f"{scalar_log}/{k}", v)

                    grad_contrib_ratios = logs["gradient_contribs"]
                    if grad_contrib and grad_contrib_ratios:
                        # Save the gradient contribution bar plot
                        plt.figure()
                        plt.title("Loss gradient contribution")
                        headers = [k.split('/')[-1][:4]
                                   for k in grad_contrib_ratios.keys()]
                        plt.bar(headers, grad_contrib_ratios.values())
                        file_path = Path("assets/artifacts/contribs/",
                                         f"{datetime.now()}.jpg")
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(file_path)
                        mlflow_manager.log_artifact(file_path)

                    output_images = logs["output_images"]
                    if save_debug and output_images:
                        grid = make_grid(output_images, nrow=8, normalize=True)
                        file_path = f"assets/artifacts/outputs/step_{global_step}.png"
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        save_image(grid, file_path)
                        mlflow_manager.log_artifact(file_path)

                global_step += 1

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
