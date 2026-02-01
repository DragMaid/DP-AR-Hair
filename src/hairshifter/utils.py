import torch
from pathlib import Path
from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt


def enabled_rely(func):
    def wrapper(self, *args, **kwargs):
        if self.enabled:
            res = func(self, *args, **kwargs)
            return res
    return wrapper


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


def enabled_rely(func):
    def wrapper(self, *args, **kwargs):
        if self.enabled:
            res = func(self, *args, **kwargs)
            return res
    return wrapper
