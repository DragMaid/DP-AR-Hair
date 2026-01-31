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


class Trainer:
    def __init__(self, args):
        self.args = args

        self.init_ddp()
        self.init_pipeline()

    def init_ddp(self):
        # TODO: allow user to select a specific device
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
            self.backend = "nccl"
        else:
            self.device = torch.device("cpu")
            self.backend = "gloo"

        dist.init_process_group(backend=self.backend)

    def init_pipeline(self):
        # TODO: Add the proper URL instead of the default localhost:5000 (PostgreSQL)
        self.mlflow_manager = MLFlowManager()

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

        self.logger = StepLogger()
        self.pipeline = TrainingPipeline(
            self.device, self.logger, self.local_rank)

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
            disc_optimizer=self.disc_optimizer
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

    def save_contrib_plot(self, grad_contrib_ratios, global_step):
        # Save the gradient contribution bar plot
        plt.figure()
        plt.title("Loss gradient contribution")
        headers = [k.split('/')[-1][:4]
                   for k in grad_contrib_ratios.keys()]
        plt.bar(headers, grad_contrib_ratios.values())
        file_path = Path("assets/artifacts/contribs/",
                         f"step_{global_step}.jpg")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)
        self.mlflow_manager.log_artifact(file_path, artifact_path="contribs")

    def save_debug_image(self, output_images, global_step):
        grid = make_grid(
            output_images, nrow=8, normalize=True)
        file_path = f"assets/artifacts/outputs/step_{global_step}.png"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, file_path)
        self.mlflow_manager.log_artifact(file_path, artifact_path="outputs")

    def train(self):
        first_processor = dist.get_rank() == 0
        with self.mlflow_manager.start_run():
            global_step = 0
            for epoch in range(self.start_epoch, self.args.epochs):
                self.sampler.set_epoch(epoch)
                epoch_iterator = tqdm(enumerate(self.dataloader), total=len(self.dataloader),
                                      desc=f"Epoch {epoch+1}/{self.args.epochs}")

                for step, batch in epoch_iterator:
                    grad_contrib = (
                        global_step+1) % pco.training.log_config.grad_contrib_interval == 0
                    param_norm = (
                        global_step+1) % pco.training.log_config.param_norm_interval == 0
                    param_dist = (
                        global_step+1) % pco.training.log_config.param_dist_interval == 0
                    save_debug = (
                        global_step+1) % pco.training.log_config.output_save_interval == 0

                    self.step(
                        batch=batch,
                        global_step=global_step,
                        first_processor=first_processor,
                        grad_contrib=grad_contrib,
                        param_dist=param_dist,
                        param_norm=param_norm,
                        save_debug=save_debug,
                    )

                    global_step += 1

        # epoch end — checkpoint
        if first_processor and ((epoch + 1) % self.args.save_weight_every) == 0:
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
        param_norm: bool,
        save_debug: bool
    ):
        # Get the 3 images: original front / side and hair transfered image
        I_s = batch["reference"]["content"]
        I_d = batch["driving"]["content"]
        I_d_dilde = batch["generated"]["content"]

        # TODO: check all of this shit
        self.pipeline.train_step(
            I_s, I_d, I_d_dilde,
            mini_batch_size=self.args.mini_batch_size,
            scaler=self.scaler,
            accumulate_grad_contrib=grad_contrib,
            store_outputs=True)

        # Log gradient norms every step
        if first_processor:
            self.logger.calculate_grad_norms(
                self.pipeline.modules_to_log)

        if first_processor and param_dist:
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
            if param_norm:
                self.logger.calculate_param_norms(
                    self.pipeline.modules_to_log)

            if param_dist:
                self.logger.log_param_distribution(
                    "generator", self.pipeline.generator_trainable_params)
                self.logger.log_param_distribution(
                    "discriminator", self.pipeline.disc_trainable_params)

        # All the final logging to management server
        if first_processor:
            logs = self.logger.finalize()

            grad_contrib_ratios = logs.pop("gradient_contribs")
            if grad_contrib and grad_contrib_ratios:
                self.save_contrib_plot(grad_contrib_ratios, global_step)

            output_images = logs.pop("output_images")
            if save_debug and output_images:
                self.save_debug_image(output_images, global_step)

            self.mlflow_manager.log_recusrive_scalar(
                log=logs, name="", step=global_step)


if __name__ == "__main__":
    args = get_args()
    trainer = Trainer(args=args)
    trainer.train()
