import torch.nn as nn
import torch.nn.init as init


class PatchGANDiscriminator(nn.Module):
    def __init__(self, n_in_channels, n_filters=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        kernel_size = 4
        pad_width = 1

        sequence = [
            nn.Conv2d(n_in_channels, n_filters,
                      kernel_size=kernel_size, stride=2, padding=pad_width),
            nn.LeakyReLU(0.2, True)
        ]

        filter_multiplier = 1
        filter_multiplier_prev = 1
        for n in range(1, n_layers):
            filter_multiplier_prev = filter_multiplier
            filter_multiplier = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(n_filters * filter_multiplier_prev,
                          n_filters * filter_multiplier,
                          kernel_size=kernel_size,
                          stride=2,
                          padding=pad_width),
                nn.BatchNorm2d(n_filters * filter_multiplier),
                nn.LeakyReLU(0.2, True)
            ]

        filter_multiplier_prev = filter_multiplier
        filter_multiplier = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(n_filters * filter_multiplier_prev,
                      n_filters * filter_multiplier,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=pad_width),
            nn.BatchNorm2d(n_filters * filter_multiplier),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(n_filters * filter_multiplier,
                               1,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=pad_width)
                     ]

        self.model = nn.Sequential(*sequence)

    def forward(self, in_image):
        return self.model(in_image)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0.2,
                             mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)
