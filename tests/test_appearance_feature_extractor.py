import torch
import pytest
from src.models.appearance_feature_extractor import AppearanceFeatureExtractor


@pytest.fixture
def appearance_extractor():
    torch.manual_seed(0)
    return AppearanceFeatureExtractor(
        block_expansion=64,
        num_resblocks=4,
        max_features=1024,
        num_down_blocks=2,
        reshape_channel=32
    )


@pytest.fixture
def sample_image():
    torch.manual_seed(0)
    return torch.randn(2, 3, 256, 256)


def test_appearance_extractor_output_shape(appearance_extractor, sample_image):
    out = appearance_extractor(sample_image)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 4


def test_appearance_extractor_no_nans(appearance_extractor, sample_image):
    out = appearance_extractor(sample_image)
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_appearance_extractor_gradients(appearance_extractor, sample_image):
    out = appearance_extractor(sample_image)
    out.sum().backward()

    for name, p in appearance_extractor.named_parameters():
        assert p.grad is not None, f"{name} has no gradient!"


def test_appearance_extractor_snapshot(appearance_extractor, sample_image, data_regression):
    out = appearance_extractor(sample_image).detach().cpu().numpy()
    data_regression.check({"output": out.tolist()})
