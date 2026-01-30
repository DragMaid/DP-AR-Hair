import torch
from collections import defaultdict


def disabled_rely(func):
    def wrapper(self, *args, **kwargs):
        if not self.disabled:
            res = func(self, *args, **kwargs)
            return res
    return wrapper


def get_grad_norm_tensor(grads: torch.tensor):
    grads = [g for g in grads if g is not None]

    if not grads:
        return 0.0

    total_norm = torch.norm(
        torch.stack([g.detach().norm(2) for g in grads]), p=2)

    return float(total_norm)


def get_grad_norm_params(params):
    params = [p for p in params if p.grad is not None]

    if not params or len(params) == 0:
        return 0.0

    total_norm = torch.norm(
        torch.stack([p.grad.detach().norm(2) for p in params]), p=2)

    return float(total_norm)


def get_param_norm(params):
    params = [p for p in params if p is not None]

    if not params:
        return 0.0

    total_norm = torch.norm(
        torch.stack([p.detach().norm(2) for p in params]), p=2
    )

    return float(total_norm)


def param_distribution_stats(params):
    params = [p.detach().view(-1) for p in params if p.requires_grad]
    if not params:
        return {}

    flat = torch.cat(params)

    return {
        "mean": flat.mean(),
        "std": flat.std(),
        "abs_mean": flat.abs().mean(),
        "max_abs": flat.abs().max(),
    }


def snapshot_params(params):
    """
    Create a detached snapshot of parameters for later comparison.
    MUST be called before optimizer.step().
    """
    return [p.detach().clone() for p in params if p.requires_grad]


def param_update_ratio(params, prev_params, eps=1e-8):
    """
    Computes ||Δθ|| / ||θ|| for a parameter group.
    Call AFTER optimizer.step().
    """
    deltas = []
    norms = []

    for p, p_prev in zip(params, prev_params):
        if not p.requires_grad:
            continue

        delta = (p.detach() - p_prev).norm(2)
        norm = p_prev.norm(2)

        deltas.append(delta)
        norms.append(norm)

    if len(deltas) == 0:
        device = params[0].device if len(params) > 0 else "cpu"
        return torch.tensor(0.0, device=device)

    ratio = (
        torch.stack(deltas).norm(2) /
        (torch.stack(norms).norm(2) + eps)
    )

    return float(ratio)


class StepLogger:
    def __init__(self, disabled=False):
        self.disabled = disabled
        self.reset()

    def reset(self):
        self.micro_steps = 0
        self.loss_sums = defaultdict(float)
        self.grad_contrib = defaultdict(float)

        self.grad_norms = {}
        self.param_norms = {}

        self.prev_params = {}
        self.param_update = {}

        self.param_dist = {}
        self._image_buffer = []

    @disabled_rely
    def accumulate_loss(self, name, value):
        self.loss_sums[name] += float(value.detach())

    @disabled_rely
    def accumulate_grad_contribution(self, name, grads):
        self.grad_contrib[name] += get_grad_norm_tensor(grads)

    @disabled_rely
    def calculate_grad_norms(self, key_to_params):
        for key, params in key_to_params.items():
            self.grad_norms[key] = get_grad_norm_params(params)

    @disabled_rely
    def calculate_param_norms(self, key_to_params):
        for key, params in key_to_params.items():
            self.param_norms[key] = get_param_norm(params)

    @disabled_rely
    def snapshot_params(self, name, params):
        self.prev_params[name] = snapshot_params(params)

    @disabled_rely
    def log_param_update(self, name, params):
        prev = self.prev_params.get(name)
        if prev is None:
            return
        self.param_update[name] = param_update_ratio(params, prev)

    @disabled_rely
    def log_param_distribution(self, name, params):
        stats = param_distribution_stats(params)
        self.param_dist[name] = {
            k: float(v.cpu())
            for k, v in stats.items()
        }

    @disabled_rely
    def log_images(self, images: torch.Tensor):
        """
        Store images during gradient accumulation.
        images: [B, C, H, W] torch tensor (in 0-1 range or raw float)
        """
        # detach and move to CPU to save GPU memory
        self._image_buffer.append(images.detach().cpu())

    @disabled_rely
    def step_done(self):
        self.micro_steps += 1

    def finalize(self):
        if self.disabled or self.micro_steps == 0:
            return {}

        losses = {k: v / self.micro_steps for k, v in self.loss_sums.items()}

        grad_contributions = {}
        for k, v in self.grad_contrib.items():
            if k != "generator_loss":
                grad_contributions[k] = v / \
                    (self.grad_contrib["generator_loss"] + 1e-8)

        all_images = torch.cat(
            self._image_buffer, dim=0) if self._image_buffer else None

        return {
            "losses": losses,
            "gradient_contribs": grad_contributions,
            "gradient_norms": self.grad_norms,
            "param_norms": self.param_norms,
            "param_update_ratios": self.param_update,
            "param_dist": self.param_dist,
            "output_images": all_images,
        }
