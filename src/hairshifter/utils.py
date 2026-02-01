import torch.nn.functional as F
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


def jitter_binary_mask(mask, p=0.5):
    """
    mask: (B, 1, H, W), values {0,1}
    p: probability to apply jitter
    """
    if torch.rand(1).item() > p:
        return mask

    # randomly choose dilation or erosion
    if torch.rand(1).item() < 0.5:
        # dilation
        return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    else:
        # erosion
        return -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)


def discriminator_augment_pair(I_d, I_p, p=0.5):
    """
    Apply augmentation to both real (I_d) and fake (I_p) in a consistent way.
    Each sample in batch is augmented or not together.
    """
    B = I_d.size(0)
    I_d_aug, I_p_aug = I_d.clone(), I_p.clone()

    for i in range(B):
        if torch.rand(1).item() < p:
            # horizontal flip
            if torch.rand(1).item() < 0.5:
                I_d_aug[i] = torch.flip(I_d_aug[i], dims=[2])
                I_p_aug[i] = torch.flip(I_p_aug[i], dims=[2])
            # mild brightness jitter
            if torch.rand(1).item() < 0.5:
                factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
                I_d_aug[i] = I_d_aug[i] * factor
                I_p_aug[i] = I_p_aug[i] * factor
            # very light Gaussian noise
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(I_d_aug[i]) * 0.01
                I_d_aug[i] = I_d_aug[i] + noise
                I_p_aug[i] = I_p_aug[i] + noise

    return I_d_aug, I_p_aug
