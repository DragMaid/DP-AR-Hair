import torch
import torch.nn as nn


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
                 hidden_channels=64,
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
        self.param_free_norm = nn.InstanceNorm2d(num_channels)

        padding = kernel_size // 2

        # MLP to produce gamma/beta from f_n
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
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


# ---------------------------
# Example usage / smoke test
# ---------------------------
# TODO: move this to test folder later
if __name__ == "__main__":
    B, C, H, W = 2, 128, 64, 64
    Cf, Cm = 64, 1  # context feature channels + mask

    h_w = torch.randn(B, C, H, W)
    h_c = torch.randn(B, C, H, W)
    f_c = torch.randn(B, Cf, H, W)
    m_c = torch.randint(0, 2, (B, Cm, H, W)).float()
    f_n = torch.cat([f_c, m_c], dim=1)

    gf_spade = GFSPADE(num_channels=C, cond_channels=Cf + Cm)
    out = gf_spade(f_n, h_c, h_w)
    print("GF-SPADE output shape:", out.shape)  # expect (B, C, H, W)
