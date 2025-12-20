import torch
import pytest
from configs.model_config import model_config
from models.context_decoder import ContextEncoder


@pytest.fixture
def context_encoder():
    return ContextEncoder(
        **model_config.context_encoder_params.model_dump())


@pytest.fixture
def sample_image():
    return torch.randn(2, 3, 256, 256)  # Input image size


def test_context_encoder_output_shape(context_encoder, sample_image):
    out = context_encoder(sample_image)
    print(out.size())
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 4


def test_context_encoder_no_nans(context_encoder, sample_image):
    out = context_encoder(sample_image)
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_context_encoder_gradients(context_encoder, sample_image):
    out = context_encoder(sample_image)
    out.sum().backward()

    for name, p in context_encoder.named_parameters():
        assert p.grad is not None, f"{name} has no gradient!"
