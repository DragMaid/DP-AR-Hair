import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from live_portrait.models.spade import SPADE


class GFSPADE(nn.Module):
    """
    GF-SPADE block as described in Hair Shifter.
    - Modulates context activation h_c using gamma/beta from f_n
    - Computes spatial gate from concat(h_w, h_c_tilde)
    - Fuses synthesis and modulated context activations
    """

    def __init__(self,
                 num_channels,
                 cond_channels,
                 hidden_channels=128,
                 kernel_size=3,
                 post_conv=True):
        """
        Args:
            num_channels: channels of x_c and x_w
            cond_channels: channels of f_n = channels(f_c) + 1 for mask
            hidden_channels: hidden channels for modulation MLP
            kernel_size: conv kernel size for modulation
            post_conv: whether to add conv+ReLU after fusion (part of ResBlock)
        """
        super().__init__()
        self.num_channels = num_channels
        self.hidden_channels = hidden_channels

        # Normalization for context activation
        self.param_free_norm = nn.InstanceNorm2d(
            num_channels, affine=False, eps=1e-5)

        padding = kernel_size // 2

        # MLP to produce gamma/beta from f_n
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=False)
        )
        self.mlp_gamma = nn.Conv2d(
            hidden_channels, num_channels, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(
            hidden_channels, num_channels, kernel_size=kernel_size, padding=padding)

        nn.init.constant_(self.mlp_gamma.weight, 0.0)
        nn.init.constant_(self.mlp_gamma.bias, 0.0)
        nn.init.constant_(self.mlp_beta.weight, 0.0)
        nn.init.constant_(self.mlp_beta.bias, 0.0)

        # Spatial gating conv
        self.gate_conv = nn.Conv2d(
            2 * num_channels, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.gate_conv.bias, -2.0)  # favor context initially
        nn.init.kaiming_normal_(self.gate_conv.weight, a=0.2)

        # Optional conv + activation after fusion
        self.post_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False)
        ) if post_conv else nn.Identity()

    def forward(self, f_n, h_c, h_w):
        """
        Args:
            f_n: concated from m_c and f_c (B,Cf,Hf,Wf)
            h_w: synthesis activation (B,C,H,W)
            h_c: context activation (B,C,H,W)
        Returns:
            fused activation h_w_tilde (B,C,H,W)
        """
        # 2. Modulate context activation
        normed = self.param_free_norm(h_c)
        actv = self.mlp_shared(f_n)
        # NOTE: opting for the conv split to save memory is good or not ?
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        h_c_tilde = normed * (1 + gamma) + beta  # Eq.1 Page 5

        # 3. Compute spatial gate
        gate_in = torch.cat([h_w, h_c_tilde], dim=1)
        m_hat = torch.sigmoid(self.gate_conv(gate_in))  # Eq.2 Page 5

        # 4. Fuse activations
        h_w_tilde = (1 - m_hat) * h_c_tilde + m_hat * h_w  # Eq.3 Page 5

        # 5. Optional post-conv + activation (part of ResBlock)
        h_w_tilde = self.post_conv(h_w_tilde)

        return h_w_tilde


class GFSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, concat_nc, use_se=False, dilation=1):
        super().__init__()

        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se

        # create conv layers
        self.conv_0 = nn.Conv2d(
            fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(
            fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

        # define GF-SPADEs
        self.gf_spade_1 = GFSPADE(fin, concat_nc)
        self.gf_spade_2 = GFSPADE(fmiddle, concat_nc)

    def forward(self, f_n, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.norm_0(x, seg1)
        dx = self.gf_spade_1(f_n, dx)
        dx = self.conv_0(self.actvn(dx))
        dx = self.norm_1(dx, seg1)
        dx = self.gf_spade_2(f_n, dx)
        dx = self.conv_1(self.actvn(dx))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
