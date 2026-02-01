import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from data.celebvhq_generated import CelebVHQGeneratedDataset
from pipelines.training_pipeline import TrainingPipeline
import torch.distributed as dist
from configs.pipeline_config import pipeline_config as pco
from utils import (
    MLFlowManager,
    save_contrib_plot,
    save_debug_image,
    save_param_histogram,
    EMA
)
from losses.utils import StepLogger


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
    p.add_argument("--mlflow", action="store_true",
                   help="Specify whether to log to mlflow server")
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
        # TODO: Add the proper URL instead of the default localhost:5000 (PostgreSQL)
        self.mlflow_manager = MLFlowManager(
            enabled=self.first_processor and self.args.mlflow)

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
            sampler=self.sampler
        )

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

        # Discrimination optmizer
        self.disc_optimizer = torch.optim.Adam(
            self.pipeline.L_adv.parameters(),
            lr=pco.training.discriminator.learn_rate,
            betas=pco.training.discriminator.betas)

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

    def train(self):
        with self.mlflow_manager.start_run():
            global_step = 0
            for epoch in range(self.start_epoch, self.args.epochs):
                self.sampler.set_epoch(epoch)
                self.ema.decay = self.initial_decay + epoch * self.decay_increase_rate
                epoch_iterator = tqdm(
                    enumerate(self.dataloader),
                    total=len(self.dataloader),
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
                    )

                    global_step += 1

        # epoch end — checkpoint
        if self.first_processor \
                and ((epoch + 1) % self.args.save_weight_every) == 0:
            ck_path = os.path.join(
                self.args.save_dir, f"epoch_{epoch+1:04d}.pt")
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
        save_debug: bool
    ):
        # Get the 3 images: original front / side and hair transfered image
        I_s = batch["reference"]["content"]
        I_d = batch["driving"]["content"]
        I_d_dilde = batch["generated"]["content"]

        self.pipeline.train_step(
            I_s, I_d, I_d_dilde,
            mini_batch_size=self.args.mini_batch_size,
            scaler=self.scaler,
            accumulate_grad_contrib=grad_contrib,
            store_outputs=save_debug)

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
        self.scaler.step(self.generator_optimizer)
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
                    save_debug_image(output_images, global_step),
                    artifact_path="outputs"
                )

            self.mlflow_manager.log_recusrive_scalar(
                log=logs, name="", step=global_step)


if __name__ == "__main__":
    args = get_args()
    trainer = Trainer(args=args)
    trainer.train()
