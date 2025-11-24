GF-SPADE (Gated Fusion SPADE)
==============================

Overview
--------
GF-SPADE is a conditional normalization and gated fusion block used in Hair Shifter-style networks.
It modulates a context activation using per-channel scale and bias (gamma/beta) predicted from a
conditioning tensor, computes a spatial gating map between the modulated context and a synthesis
activation, and fuses them.

Key behavior
------------
- Normalizes context activation with InstanceNorm2d.
- Uses a small convolutional MLP to predict gamma and beta from the conditioning tensor f_n.
- Applies modulation: h_c_tilde = norm(h_c) * (1 + gamma) + beta
- Computes a spatial gate m_hat = sigmoid(conv(concat(h_w, h_c_tilde))).
- Produces fused activation: (1 - m_hat) * h_c_tilde + m_hat * h_w
- Optionally applies a post-convolution + ReLU.

Constructor
-----------
GFSPADE(num_channels, cond_channels, hidden_channels=64, kernel_size=3, post_conv=True)

- num_channels: number of channels in the synthesis/context activations (h_w and h_c).
- cond_channels: number of channels in the conditioning tensor f_n (e.g., context features + mask channel).
- hidden_channels: hidden channels used inside the modulation MLP.
- kernel_size: convolution kernel size used in the modulation MLP.
- post_conv: whether to apply a conv+ReLU after fusion (useful when the block is part of a ResBlock).

Usage example
-------------
- Create the block with matching channel sizes, call it with f_n, h_c, h_w where all spatial sizes match.
- f_n is typically a concatenation of context features and a mask channel.

Notes
-----
- The gating conv bias is initialized to -2 to bias the block towards using the context initially.
- The gamma/beta prediction convs are initialized to zero so the modulation starts as identity.

References
----------
- Implementation based on the GF-SPADE description from Hair Shifter.

