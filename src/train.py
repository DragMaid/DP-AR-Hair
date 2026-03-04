import os
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from data.celebvhq_generated import CelebVHQGeneratedDataset
from pipelines.training_pipeline import TrainingPipeline
import torch.distributed as dist
from configs.pipeline_config import pipeline_config as pco
from hairshifter.mlflow_manager import MLFlowManager
from hairshifter.ema import EMA
from hairshifter.utils import (
    save_contrib_plot,
    save_debug_image,
    save_param_histogram
)
from losses.utils import StepLogger


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed: int = 42):
    # Python
    random.seed(seed)

    # Environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] Everything seeded with seed={seed}")


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
    # TODO: change this to steps instead
    p.add_argument("--save_per_steps", type=int, help="save weight every N steps",
                   default=pco.training.steps_till_save)
    p.add_argument("--freeze_disc", type=int,
                   help="freeze discriminator for N steps", default=0)
    p.add_argument("--freeze_gen", type=int,
                   help="freeze generator for N steps", default=0)
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to resume")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="Specify the mlflow server to log to")
    p.add_argument("--warmup_disc", type=int, default=0,
                   help="Specify discriminator warmup steps to perform")
    p.add_argument("--warmup_gen", type=int, default=0,
                   help="Specify generator warmup steps to perform")
    p.add_argument("--seed", type=int, default=None,
                   help="Specify seed when initilizing all modules")
    # p.add_argument("--device", type=str, default=None)
    p.add_argument("--mixed_precision", action="store_true")
    return p.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args

        self.init_ddp()
        self.init_pipeline()

    def init_ddp(self):
        # TODO: allow user to select a specific device
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.first_processor = self.local_rank == 0

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.backend = "nccl"
        else:
            self.device = torch.device("cpu")
            self.backend = "gloo"

        dist.init_process_group(backend=self.backend)

    def init_pipeline(self):
        self.mlflow_manager = MLFlowManager(
            uri=self.args.mlflow_uri if self.args.mlflow_uri else "http://localhost:5000",
            enabled=self.first_processor and self.args.mlflow_uri)

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

        self.dataset = CelebVHQGeneratedDataset(
            dataset_dir=self.args.dataset,
            transform=transform
        )

        self.sampler = DistributedSampler(self.dataset)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True, drop_last=True,
            sampler=self.sampler,
            worker_init_fn=seed_worker
        )
        self.steps_count = len(self.dataloader)

        self.logger = StepLogger(enabled=self.first_processor)
        self.pipeline = TrainingPipeline(
            self.device, self.logger, self.local_rank)

        # TODO: change this to config decay rate
        self.initial_decay = 0.99
        self.final_decay = 0.999
        self.decay_increase_rate = abs(
            self.final_decay - self.initial_decay) / self.args.epochs
        self.ema = EMA(self.pipeline.generator_trainable_dict,
                       self.initial_decay, enabled=self.first_processor)

        # Refering to entire pipeline beside IIHT
        self.generator_optimizer = torch.optim.Adam(
            self.pipeline.generator_trainable_params,
            lr=pco.training.generator.learn_rate,
            betas=pco.training.generator.betas)
        self.generator_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.generator_optimizer, lr_lambda=lambda cs: self.warmup(cs, self.args.warmup_gen))

        # Discrimination optmizer
        self.disc_optimizer = torch.optim.Adam(
            self.pipeline.L_adv.parameters(),
            lr=pco.training.discriminator.learn_rate,
            betas=pco.training.discriminator.betas)
        self.disc_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.disc_optimizer, lr_lambda=lambda cs: self.warmup(cs, self.args.warmup_disc))

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.args.mixed_precision and self.device.type == "cuda")

        # Set the optimizers to be used in pipeline
        self.pipeline.set_optimizers(
            generator_optimizer=self.generator_optimizer,
            disc_optimizer=self.disc_optimizer,
            ema=self.ema
        )

        self.start_epoch = 0
        if self.args.resume:
            ck = self.pipeline.load_checkpoint(
                self.args.resume, load_optimizers=True)
            if "epoch" in ck:
                self.start_epoch = ck["epoch"] + 1
            print(
                f"Resumed from {self.args.resume}, start_epoch={self.start_epoch}")

        os.makedirs(self.args.save_dir, exist_ok=True)

    def warmup(self, current_step, steps):
        if current_step < steps:
            return float(current_step) / float(max(1, steps))
        else:
            return 1.0

    def train(self):
        with self.mlflow_manager.start_run():
            global_step = 0
            for epoch in range(self.start_epoch, self.args.epochs):
                self.sampler.set_epoch(epoch)
                self.ema.decay = self.initial_decay + epoch * self.decay_increase_rate
                epoch_iterator = tqdm(
                    enumerate(self.dataloader),
                    total=self.steps_count,
                    desc=f"Epoch {epoch+1}/{self.args.epochs}"
                )

                for step, batch in epoch_iterator:
                    grad_contrib = (
                        global_step+1) % pco.training.log_config.grad_contrib_interval == 0
                    param_norm = (
                        global_step+1) % pco.training.log_config.param_norm_interval == 0
                    param_dist = (
                        global_step+1) % pco.training.log_config.param_dist_interval == 0
                    param_ratio = (
                        global_step+1) % pco.training.log_config.param_ratio_interval == 0
                    param_hist = (
                        global_step+1) % pco.training.log_config.param_hist_interval == 0
                    save_debug = (
                        global_step+1) % pco.training.log_config.output_save_interval == 0
                    ema_update = (
                        global_step+1) % pco.training.log_config.ema_update_interval == 0
                    freeze_disc = (global_step + 1) <= self.args.freeze_disc
                    freeze_gen = (global_step + 1) <= self.args.freeze_gen

                    self.step(
                        batch=batch,
                        global_step=global_step,
                        first_processor=self.first_processor,
                        grad_contrib=grad_contrib,
                        param_dist=param_dist,
                        param_ratio=param_ratio,
                        param_hist=param_hist,
                        param_norm=param_norm,
                        ema_update=ema_update,
                        save_debug=save_debug,
                        freeze_disc=freeze_disc,
                        freeze_gen=freeze_gen,
                    )

                    global_step += 1

                    # epoch end — checkpoint
                    is_save_step = (
                        global_step + 1) % self.args.save_per_steps == 0
                    is_last_step = global_step + 1 == self.args.epochs * self.steps_count
                    if self.first_processor and (is_save_step or is_last_step):
                        date = datetime.now().strftime('%Y-%m-%d_%H:%M')
                        ck_path = os.path.join(
                            self.args.save_dir, f"epoch_{epoch+1:04d}_{date}.pt")
                        self.pipeline.save_checkpoint(ck_path, epoch=epoch)
                        print(f"Saved checkpoint: {ck_path}")

        dist.destroy_process_group()
        print("Training complete.")

    def step(
        self,
        batch: dict,
        global_step: int,
        first_processor: bool,
        grad_contrib: bool,
        param_dist: bool,
        param_ratio: bool,
        param_hist: bool,
        param_norm: bool,
        ema_update: bool,
        save_debug: bool,
        freeze_disc: bool,
        freeze_gen: bool
    ):
        # Get the 3 images: original front / side and hair transfered image
        I_s = batch["reference"]["content"]
        I_d = batch["driving"]["content"]
        I_d_dilde = batch["generated"]["content"]

        self.pipeline.train_step(
            I_s, I_d, I_d_dilde,
            mini_batch_size=self.args.mini_batch_size,
            scaler=self.scaler,
            accumulate_grad_contrib=grad_contrib and self.first_processor,
            store_outputs=save_debug and self.first_processor,
            freeze_discriminator=freeze_disc,
            freeze_generator=freeze_gen)

        # Log gradient norms every step
        if first_processor:
            self.logger.calculate_grad_norms(
                self.pipeline.modules_to_log)

        # Creating first snapshot to calculate update ratio
        if first_processor and param_ratio:
            self.logger.snapshot_params(
                "generator", self.pipeline.generator_trainable_params)
            self.logger.snapshot_params(
                "discriminator", self.pipeline.disc_trainable_params)

        # Update the weights and reset optimizer
        self.scaler.step(self.disc_optimizer)
        self.disc_scheduler.step()

        self.scaler.step(self.generator_optimizer)
        self.generator_scheduler.step()

        self.disc_optimizer.zero_grad(set_to_none=True)
        self.generator_optimizer.zero_grad(set_to_none=True)

        self.scaler.update()

        if first_processor:
            # Call exponential moving average update
            if ema_update:
                self.ema.update()

            if param_norm:
                self.logger.calculate_param_norms(
                    self.pipeline.modules_to_log)

            if param_dist:
                # Log parameter distribution
                self.logger.log_param_distribution(
                    "generator", self.pipeline.generator_trainable_params)
                self.logger.log_param_distribution(
                    "discriminator", self.pipeline.disc_trainable_params)

            if param_ratio:
                # Log parameter update rate
                self.logger.log_param_update(
                    "generator", self.pipeline.generator_trainable_params)
                self.logger.log_param_update(
                    "discriminator", self.pipeline.disc_trainable_params)

            if param_hist:
                self.mlflow_manager.log_artifact(
                    save_param_histogram(
                        params=self.pipeline.generator_trainable_params,
                        global_step=global_step
                    ),
                    artifact_path="hist"
                )

        # All the final logging to management server
        if first_processor:
            logs = self.logger.finalize()

            grad_contrib_ratios = logs.pop("gradient_contribs")
            if grad_contrib and grad_contrib_ratios:
                self.mlflow_manager.log_artifact(
                    save_contrib_plot(grad_contrib_ratios, global_step),
                    artifact_path="contribs"
                )

            output_images = logs.pop("output_images")
            if save_debug and output_images is not None:
                self.mlflow_manager.log_artifact(
                    save_debug_image(
                        source=I_s,
                        driving=I_d_dilde,
                        output=output_images,
                        global_step=global_step),
                    artifact_path="outputs"
                )

            self.mlflow_manager.log_recusrive_scalar(
                log=logs, name="", step=global_step)


if __name__ == "__main__":
    args = get_args()
    if args.seed:
        seed_everything(args.seed)
    trainer = Trainer(args=args)
    trainer.train()
