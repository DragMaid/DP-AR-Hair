# MSGSpadeDecoder
Location: `src/models/msg_spade_decoder.py`.

## Overview
`MSGSpadeDecoder` is a compositional decoder that combines a Context Decoder and a Synthesis Decoder (SPADE-style). It orchestrates multiple middle residual blocks to progressively merge context and synthesis features and produces a final RGB image via sigmoid activation.

## Constructor
The constructor accepts two decoder instances:
- `context_decoder`: an instance of `ContextDecoder` (or compatible object) exposed as `D_C`.
- `synthesis_decode`: an instance of `SynthesisDecoder` (or compatible object) exposed as `D_S`.

## Forward
The `forward` method expects three tensors:
- `f_c` (torch.Tensor): context features, shape (B, C, H, W).
- `f_w` (torch.Tensor): synthesis features, shape (B, C, H, W).
- `m_c` (torch.Tensor): motion/context features to be concatenated with `f_c`, shape (B, C_m, H, W).
- Returns a single RGB image tensor `fw` of shape (B, 3, H, W) with values in [0, 1].

## Behavior
The decoder performs the following high-level steps:
1. Concatenate `m_c` and `f_c` into `f_n`.
2. Compute initial feature maps using `D_C.fc` and `D_S.fc`.
3. Loop through six middle residual blocks `G_middle_0`..`G_middle_5`:
   - Run context-block operations (shortcut, norms, activations, convs).
   - Run synthesis-block operations with gated fusion (`gf_spade_1`, `gf_spade_2`) that mix context and synthesis.
   - Sum residual shortcuts with processed activations to form next-level features.
4. Apply upsampling and final convs on both decoders to produce images and apply sigmoid.

## Testing
A simple unit test exists under `tests/test_msg_spade_decoder.py`. The test constructs minimal dummy decoders that provide the attributes/methods expected by `MSGSpadeDecoder` and checks that a forward pass returns a tensor of shape `(B, 3, H, W)` and values in `[0, 1]`.

## Notes
- This module assumes its decoder inputs expose specific method names (e.g., `G_middle_{i}`, `fc`, `up`, `conv_img`, etc.).
- If integrating with the project's real `ContextDecoder` and `SynthesisDecoder`, ensure their API matches the expectations.
