import torch
import pytest
from models.spade import SPADE, SPADEResnetBlock


# ---------------------------------------------------------
# SPADE TESTS
# ---------------------------------------------------------

def test_spade_output_shape():
    x = torch.randn(2, 64, 32, 32)
    seg = torch.randn(2, 20, 16, 16)

    spade = SPADE(norm_nc=64, label_nc=20)
    out = spade(x, seg)

    assert out.shape == x.shape, "SPADE output shape must match input shape"


def test_spade_gamma_beta_shapes():
    x = torch.randn(2, 64, 32, 32)
    seg = torch.randn(2, 20, 32, 32)
    spade = SPADE(norm_nc=64, label_nc=20)

    # Run forward to trigger gamma/beta creation
    with torch.no_grad():
        _ = spade(x, seg)

    # Extract gamma and beta via forward hooks
    gamma_beta = {}

    def hook_gamma(module, inp, out):
        gamma_beta['gamma'] = out.clone()

    def hook_beta(module, inp, out):
        gamma_beta['beta'] = out.clone()

    g_hook = spade.mlp_gamma.register_forward_hook(hook_gamma)
    b_hook = spade.mlp_beta.register_forward_hook(hook_beta)

    _ = spade(x, seg)

    g_hook.remove()
    b_hook.remove()

    assert gamma_beta['gamma'].shape == x.shape, "gamma must match input spatial shape"
    assert gamma_beta['beta'].shape == x.shape, "beta must match input spatial shape"


def test_spade_backward():
    x = torch.randn(1, 64, 16, 16, requires_grad=True)
    seg = torch.randn(1, 20, 16, 16)

    spade = SPADE(norm_nc=64, label_nc=20)
    out = spade(x, seg)

    out.mean().backward()

    # Check that gradients exist
    assert spade.mlp_gamma.weight.grad is not None
    assert spade.mlp_beta.weight.grad is not None


# ---------------------------------------------------------
# SPADEResnetBlock TESTS
# ---------------------------------------------------------

@pytest.mark.parametrize("fin,fout", [(64, 64), (64, 128)])
def test_spaderesnetblock_output_shape(fin, fout):
    x = torch.randn(2, fin, 32, 32)
    seg = torch.randn(2, 20, 32, 32)

    block = SPADEResnetBlock(
        fin=fin, fout=fout, norm_G="instance", label_nc=20)
    out = block(x, seg)

    assert out.shape == (2, fout, 32, 32)


def test_spaderesnetblock_shortcut_identity():
    x = torch.randn(2, 64, 32, 32)
    seg = torch.randn(2, 20, 32, 32)

    block = SPADEResnetBlock(64, 64, norm_G="instance", label_nc=20)

    # fin == fout → shortcut is identity
    x_s = block.shortcut(x, seg)
    assert torch.allclose(
        x_s, x), "Shortcut should be identity when fin == fout"


def test_spaderesnetblock_shortcut_learned():
    x = torch.randn(2, 64, 32, 32)
    seg = torch.randn(2, 20, 32, 32)

    block = SPADEResnetBlock(64, 128, norm_G="instance", label_nc=20)

    x_s = block.shortcut(x, seg)
    assert x_s.shape == (
        2, 128, 32, 32), "Shortcut should use conv_s when fin != fout"


def test_spaderesnetblock_spectral_norm():
    block = SPADEResnetBlock(64, 64, norm_G="spectral", label_nc=20)

    # spectral norm adds weight_u, weight_v attributes
    assert hasattr(
        block.conv_0, 'weight_u'), "Spectral norm not applied to conv_0"
    assert hasattr(
        block.conv_1, 'weight_u'), "Spectral norm not applied to conv_1"


def test_spaderesnetblock_backward():
    x = torch.randn(1, 64, 16, 16, requires_grad=True)
    seg = torch.randn(1, 20, 16, 16)

    block = SPADEResnetBlock(64, 128, norm_G="spectral", label_nc=20)

    out = block(x, seg)
    out.mean().backward()

    # Check ALL learnable parameters received gradients
    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
