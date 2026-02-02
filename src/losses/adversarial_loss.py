import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.parametrizations import spectral_norm


class PatchGANDiscriminator(nn.Module):
    def __init__(self, n_in_channels, n_filters=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        kernel_size = 4
        pad_width = 1

        sequence = [
            spectral_norm(nn.Conv2d(n_in_channels, n_filters,
                                    kernel_size=kernel_size, stride=2, padding=pad_width)),
            nn.LeakyReLU(0.2, False)
        ]

        filter_multiplier = 1
        filter_multiplier_prev = 1
        for n in range(1, n_layers):
            filter_multiplier_prev = filter_multiplier
            filter_multiplier = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(n_filters * filter_multiplier_prev,
                                        n_filters * filter_multiplier,
                                        kernel_size=kernel_size,
                                        stride=2,
                                        padding=pad_width)),
                # nn.BatchNorm2d(n_filters * filter_multiplier),
                nn.LeakyReLU(0.2, False)
            ]

        filter_multiplier_prev = filter_multiplier
        filter_multiplier = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(n_filters * filter_multiplier_prev,
                                    n_filters * filter_multiplier,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=pad_width)),
            # nn.BatchNorm2d(n_filters * filter_multiplier),
            nn.LeakyReLU(0.2, False)
        ]

        sequence += [spectral_norm(nn.Conv2d(n_filters * filter_multiplier,
                                             1,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             padding=pad_width))]

        self.model = nn.Sequential(*sequence)

    def forward(self, in_image):
        return self.model(in_image)


def weights_init(m: nn.Module, final_layer_names=("shift", "output")):
    """
    Initialize Hair Shifter generator weights safely.

    Rules:
    - Hidden layers (Conv2d, Linear):
        - LeakyReLU activations → Kaiming normal (fan_in, a=0.2)
    - Final shift layers → small Gaussian N(0, 0.02)
    - Bias → zero
    - Handles spectral_norm (init weight_orig)
    - Optionally, uses layer names to identify final shift layer
    """

    # Identify spectral norm layers
    if hasattr(m, "weight_orig"):
        weight_tensor = m.weight_orig
    elif hasattr(m, "weight"):
        weight_tensor = m.weight
    else:
        weight_tensor = None

    # Determine if this is a final shift/output layer
    is_final = any(n in getattr(m, "name", "") for n in final_layer_names)

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if is_final:
            # Final shift layer → small Gaussian
            init.normal_(weight_tensor, mean=0.0, std=0.02)
        else:
            # Hidden layers → Kaiming for LeakyReLU
            init.kaiming_normal_(weight_tensor, a=0.2,
                                 mode="fan_in", nonlinearity="leaky_relu")

        # Bias
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias, 0.0)

    elif isinstance(m, nn.BatchNorm2d):
        if weight_tensor is not None:
            init.normal_(weight_tensor, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias, 0.0)
