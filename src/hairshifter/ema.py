import torch
from hairshifter.utils import enabled_rely


class EMA:
    def __init__(self, param_dict, decay, enabled=True):
        """
        Initialize EMA class to manage exponential moving average of model parameters.

        Args:
            model (torch.nn.Module): The model for which EMA will track parameters.
            decay (float): Decay rate, typically a value close to 1, e.g., 0.999.
        """
        self.enabled = enabled

        if not enabled:
            return

        self.param_dict = param_dict
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Store initial parameters
        with torch.no_grad():
            for name, param in self.param_dict.items():
                if param.requires_grad:
                    self.shadow[name] = param.detach().cpu().clone()

    @enabled_rely
    @torch.no_grad()
    def update(self):
        """
        Update EMA weights on CPU.
        """
        for name, param in self.param_dict.items():
            if not param.requires_grad:
                continue

            # Move current param to CPU temporarily
            param_cpu = param.detach().cpu()

            self.shadow[name].mul_(self.decay)
            self.shadow[name].add_(param_cpu, alpha=1.0 - self.decay)

    @enabled_rely
    @torch.no_grad()
    def apply_shadow(self):
        """
        Replace model params with EMA weights.
        """
        self.backup.clear()

        for name, param in self.param_dict.items():
            if not param.requires_grad:
                continue

            # Backup original param to CPU
            self.backup[name] = param.detach().cpu().clone()

            # Copy EMA weight to GPU param
            param.data.copy_(self.shadow[name].to(param.device))

    @enabled_rely
    @torch.no_grad()
    def restore(self):
        """
        Restore original model parameters from CPU backup.
        """
        for name, param in self.param_dict.items():
            if not param.requires_grad:
                continue

            param.data.copy_(self.backup[name].to(param.device))

        self.backup.clear()
