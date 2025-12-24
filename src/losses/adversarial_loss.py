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


def weights_init(m: nn.Module):
    # SpectralNorm wraps weights as `weight_orig`
    if isinstance(m, nn.Conv2d):
        if hasattr(m, "weight_orig"):
            # Spectral-normalized Conv2d
            init.kaiming_normal_(
                m.weight_orig,
                a=0.2,
                mode="fan_in",
                nonlinearity="leaky_relu",
            )
        else:
            # Plain Conv2d (in case you ever keep one)
            init.kaiming_normal_(
                m.weight,
                a=0.2,
                mode="fan_in",
                nonlinearity="leaky_relu",
            )

        if m.bias is not None:
            init.constant_(m.bias, 0.0)
