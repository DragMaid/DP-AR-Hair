from torch import nn
from live_portrait.models.utils import SameBlock2d, DownBlock2d


class ContextEncoder(nn.Module):
    """
    Non-hair Context Encoder (E_C)

    - Pure 2D encoder
    - No volumetric reshape
    - Outputs spatial context features for SPADE / MSG decoder
    - Output shape: B × 256 × 64 × 64
    """

    def __init__(self,
                 image_channel: int,
                 block_expansion: int,
                 num_down_blocks: int,
                 max_features: int,
                 out_channels: int = 256):
        super().__init__()

        # Initial conv
        self.first = SameBlock2d(
            image_channel,
            block_expansion,
            kernel_size=(3, 3),
            padding=(1, 1)
        )

        # Downsampling blocks
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(
                DownBlock2d(
                    in_features,
                    out_features,
                    kernel_size=(3, 3),
                    padding=(1, 1)
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)

        # Different from E_H that there's no 3D resblocks
        # Projection to decoder-compatible channels
        self.proj = nn.Sequential(
            nn.Conv2d(
                out_features,
                max_features,
                kernel_size=1,
                stride=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                max_features,
                out_channels,
                kernel_size=1,
                stride=1
            )
        )

    def forward(self, x):
        x = self.first(x)  # Bx3x256x256
        for block in self.down_blocks:
            x = block(x)

        f_c = self.proj(x)
        return f_c
