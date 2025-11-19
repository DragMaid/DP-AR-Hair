from torch import nn
import torch.nn.functional as F
import torch
import math
import warnings


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp

    coordinate_grid = make_coordinate_grid(spatial_size, mean)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, ref, **kwargs):
    d, h, w = spatial_size
    x = torch.arange(w).type(ref.dtype).to(ref.device)
    y = torch.arange(h).type(ref.dtype).to(ref.device)
    z = torch.arange(d).type(ref.dtype).to(ref.device)

    # NOTE: must be right-down-in
    x = (2 * (x / (w - 1)) - 1)  # the x axis faces to the right
    y = (2 * (y / (h - 1)) - 1)  # the y axis faces to the bottom
    z = (2 * (z / (d - 1)) - 1)  # the z axis faces to the inner

    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat(
        [xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding)

        self.conv2 = nn.Conv3d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding)

        self.norm1 = nn.BatchNorm3d(in_features, affine=True)
        self.norm2 = nn.BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock3d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=3,
                 padding=1,
                 groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              padding=padding,
                              groups=groups)
        self.norm = nn.BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=3,
                 padding=1,
                 groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              padding=padding,
                              groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=3,
                 padding=1,
                 groups=1):
        super(DownBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              padding=padding,
                              groups=groups)
        self.norm = nn.BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 groups=1,
                 kernel_size=3,
                 padding=1,
                 lrelu=False):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              padding=padding,
                              groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self,
                 block_expansion,
                 in_features,
                 num_blocks=3,
                 max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            # Compute channel sizes for this stage
            in_channels = in_features if i == 0 else min(
                max_features, block_expansion * (2 ** i))
            out_channels = min(max_features, block_expansion * (2 ** (i + 1)))

            down_blocks.append(
                DownBlock3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self,
                 block_expansion,
                 in_features,
                 num_blocks=3,
                 max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * \
                min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(
                UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

        self.conv = nn.Conv3d(in_channels=self.out_filters,
                              out_channels=self.out_filters,
                              kernel_size=3,
                              padding=1)
        self.norm = nn.BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self,
                 block_expansion,
                 in_features,
                 num_blocks=3,
                 max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(
            block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(
            block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x,
                                self.normalized_shape,
                                self.weight,
                                self.bias,
                                self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low_cdf = norm_cdf((a - mean) / std)
        up_cdf = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low_cdf - 1, 2 * up_cdf - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob=0., training=False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    This is the same as the DropConnect implementation
    I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is
    a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    I've opted for changing the layer and argument names to 'drop path'
    rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
