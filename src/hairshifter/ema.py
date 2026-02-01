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

        if enabled:
            self.param_dict = param_dict
            self.decay = decay
            self.shadow = {}
            self.backup = {}

            # Store initial parameters
            for name, param in self.param_dict.items():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()

    @enabled_rely
    def update(self):
        """
        Update shadow parameters with exponential decay.
        """
        for name, param in self.param_dict.items():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + \
                    self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    @enabled_rely
    def apply_shadow(self):
        """
        Apply shadow (EMA) parameters to model.
        """
        if not self.enabled:
            return

        for name, param in self.param_dict.items():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    @enabled_rely
    def restore(self):
        """
        Restore original model parameters from backup.
        """
        for name, param in self.param_dict.items():
            if param.requires_grad:
                param.data = self.backup[name]
