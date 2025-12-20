from torch import nn
from models.gated_fusion_spade import GFSPADEResnetBlock


class SynthesisDecoder(nn.Module):
    def __init__(self,
                 upscale=1,
                 max_features=256,
                 block_expansion=64,
                 out_channels=64,
                 num_down_blocks=2):
        input_channels = min(
            max_features,
            block_expansion * (2 ** num_down_blocks)
        )

        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels
        concat_num_channels = input_channels + 1  # 256 + 1 for hair mask

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
        self.G_middle_0 = GFSPADEResnetBlock(
            2 * input_channels, 2 * input_channels, norm_G, label_num_channels, concat_num_channels)
        self.G_middle_1 = GFSPADEResnetBlock(
            2 * input_channels, 2 * input_channels, norm_G, label_num_channels, concat_num_channels)
        self.G_middle_2 = GFSPADEResnetBlock(
            2 * input_channels, 2 * input_channels, norm_G, label_num_channels, concat_num_channels)
        self.G_middle_3 = GFSPADEResnetBlock(
            2 * input_channels, 2 * input_channels, norm_G, label_num_channels, concat_num_channels)
        self.G_middle_4 = GFSPADEResnetBlock(
            2 * input_channels, 2 * input_channels, norm_G, label_num_channels, concat_num_channels)
        self.G_middle_5 = GFSPADEResnetBlock(
            2 * input_channels, 2 * input_channels, norm_G, label_num_channels, concat_num_channels)
        self.up_0 = GFSPADEResnetBlock(
            2 * input_channels, input_channels, norm_G, label_num_channels, concat_num_channels)
        self.up_1 = GFSPADEResnetBlock(
            input_channels, out_channels, norm_G, label_num_channels, concat_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)
            )
