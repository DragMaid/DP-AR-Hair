import torch
import pytest
from configs.configs import config
from models.appearance_feature_extractor import AppearanceFeatureExtractor


@pytest.fixture
def appearance_extractor():
    return AppearanceFeatureExtractor(
        **config.appearance_feature_extractor_params.model_dump())


@pytest.fixture
def sample_image():
    return torch.randn(2, 3, 256, 256)  # Input image size


def test_appearance_extractor_output_shape(appearance_extractor, sample_image):
    out = appearance_extractor(sample_image)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 5


def test_appearance_extractor_no_nans(appearance_extractor, sample_image):
    out = appearance_extractor(sample_image)
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_appearance_extractor_gradients(appearance_extractor, sample_image):
    out = appearance_extractor(sample_image)
    out.sum().backward()

    for name, p in appearance_extractor.named_parameters():
        assert p.grad is not None, f"{name} has no gradient!"


def test_appearance_extractor_snapshot(appearance_extractor,
                                       sample_image,
                                       data_regression):
    out = appearance_extractor(sample_image).detach().cpu().numpy()
    data_regression.check({"output": out.tolist()})
