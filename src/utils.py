import mlflow
import logging
import torch
from contextlib import nullcontext
from pathlib import Path
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt


def save_contrib_plot(grad_contrib_ratios, global_step):
    plt.figure()
    plt.title("Loss gradient contribution")
    headers = [k.split('/')[-1][:4]
               for k in grad_contrib_ratios.keys()]
    plt.bar(headers, grad_contrib_ratios.values())
    file_path = Path("assets/artifacts/contribs/",
                     f"step_{global_step}.jpg")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_path)
    return file_path


def save_debug_image(output_images, global_step):
    grid = make_grid(
        output_images, nrow=8, normalize=True)
    file_path = Path(f"assets/artifacts/outputs/step_{global_step}.png")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, file_path)
    return file_path


def decoder_param_filter(params):
    return max(
        (p for p in params if p.requires_grad),
        key=lambda p: p.numel()
    )


@torch.no_grad()
def save_param_histogram(params, global_step, bins=100):
    save_dir = Path("assets/artifacts/histograms")
    save_dir.mkdir(parents=True, exist_ok=True)

    selected_param = decoder_param_filter(params)
    data = selected_param.detach().cpu().flatten().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.title("Parameter Histogram")
    plt.xlabel("Value")
    plt.ylabel("Count")

    filename = save_dir / \
        f"step_{global_step}.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return filename


class MLFlowManager:
    """
    Manages MLFlow experiment tracking.
    """

    def __init__(self,
                 uri: str = "http://localhost:5000",
                 enabled=True,
                 experiment_name: str = "dp-hair-training"):
        self.uri = uri
        self.enabled = enabled
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def start_run(self):
        """
        Returns a context manager.
        - If enabled: real MLflow run
        - If disabled: no-op context manager
        """
        if not self.enabled:
            return nullcontext()

        # Optional safety: avoid accidental nesting
        if mlflow.active_run() is not None:
            return nullcontext()

        return mlflow.start_run()

    def _setup_mlflow(self):
        """Configure MLFlow tracking URI and experiment."""
        if self.enabled:
            mlflow.set_tracking_uri(self.uri)
            mlflow.set_experiment(self.experiment_name)
            logging.info(f"MLflow Tracking Server is running at {self.uri}")

    def log_metric(self, *args, **kwargs):
        if self.enabled:
            mlflow.log_metric(*args, **kwargs)

    def log_artifact(self, *args, **kwargs):
        if self.enabled:
            mlflow.log_artifact(*args, **kwargs)

    def log_recusrive_scalar(self, log: dict, name: str, step: int):
        for k, v in log.items():
            appended_name = f"{name}/{k}" if name else k
            if isinstance(v, dict):
                self.log_recusrive_scalar(log=v, name=appended_name, step=step)
            else:
                self.log_metric(
                    key=appended_name, value=v, step=step)


if __name__ == "__main__":
    from time import sleep
    from random import uniform

    mlflow_manager = MLFlowManager(enabled=True)

    def generate_random_log():
        return {
            'losses': {
                'discriminator_loss': uniform(0, 100),
                'generator_loss': uniform(0, 100),
                'perceptual_loss': uniform(0, 100),
                'hair_loss': uniform(0, 100),
                'face_loss': uniform(0, 100),
                'global_loss': uniform(0, 100),
                'adversarial_gen_loss': uniform(0, 100),
            },
            'gradient_norms': {
                'generator': uniform(0, 100),
                'discriminator': uniform(0, 100),
                'context_encoder': uniform(0, 100),
                'synthesize_decoder': uniform(0, 100),
            },
            'param_norms': {
                'generator': uniform(0, 100),
                'discriminator': uniform(0, 100),
                'context_encoder': uniform(0, 100),
                'synthesize_decoder': uniform(0, 100),
            },
            'param_update_ratios': {
                'generator': uniform(0, 100),
                'discriminator': uniform(0, 100),
            },
            'param_dist': {
                'generator': {
                    'mean': uniform(0, 100),
                    'std': uniform(0, 100),
                    'abs_mean': uniform(0, 100),
                    'max_abs': uniform(0, 100),
                },
                'discriminator': {
                    'mean': uniform(0, 100),
                    'std': uniform(0, 100),
                    'abs_mean': uniform(0, 100),
                    'max_abs': uniform(0, 100),
                }
            }
        }

    with mlflow_manager.start_run():
        step = 0
        while True:
            step += 1
            log = generate_random_log()
            mlflow_manager.log_recusrive_scalar(log=log, name="", step=step)
            mlflow_manager.log_artifact(
                "assets/artifacts/contribs/test.jpg", artifact_path="contribs")
            mlflow_manager.log_artifact(
                "assets/artifacts/outputs/step_test.png", artifact_path="outputs")
            sleep(3)
