import torch
import pytest
from configs.configs import config
from models.context_decoder import ContextDecoder


@pytest.fixture
def context_decoder():
    torch.manual_seed(0)
    return ContextDecoder(
        **config.context_decoder_params.model_dump())


@pytest.fixture
def sample_feature():
    torch.manual_seed(0)
    return torch.randn(2, 256, 64, 64)


def test_context_decoder_output_shape(context_decoder, sample_feature):
    out = context_decoder(sample_feature)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 4
    assert out.shape[0] == 2  # batch size
    assert out.shape[1] == 3  # output channels (RGB)


def test_context_decoder_output_range(context_decoder, sample_feature):
    out = context_decoder(sample_feature)
    assert out.min() >= -1.0 and out.max() <= 1.0, "Output should be normalized"


def test_context_decoder_no_nans(context_decoder, sample_feature):
    out = context_decoder(sample_feature)
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_context_decoder_gradients(context_decoder, sample_feature):
    out = context_decoder(sample_feature)
    out.sum().backward()

    for name, p in context_decoder.named_parameters():
        assert p.grad is not None, f"{name} has no gradient!"


def test_context_decoder_snapshot(context_decoder, sample_feature, data_regression):
    out = context_decoder(sample_feature).detach().cpu().numpy()
    data_regression.check({"output": out.tolist()})
