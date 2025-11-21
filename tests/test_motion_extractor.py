import torch
import pytest
from src.models.motion_extractor import MotionExtractor


@pytest.fixture
def motion_extractor():
    torch.manual_seed(0)
    return MotionExtractor(
        block_expansion=64,
        num_blocks=4,
        max_features=1024,
        num_down_blocks=2,
        num_kp=10
    )


@pytest.fixture
def sample_image():
    torch.manual_seed(0)
    return torch.randn(2, 3, 256, 256)


def test_motion_extractor_output_structure(motion_extractor, sample_image):
    out = motion_extractor(sample_image)
    assert isinstance(out, dict)
    assert 'kp_value' in out
    assert isinstance(out['kp_value'], torch.Tensor)


def test_motion_extractor_kp_shape(motion_extractor, sample_image):
    out = motion_extractor(sample_image)
    kp = out['kp_value']
    assert kp.ndim == 3
    assert kp.shape[0] == 2  # batch size
    assert kp.shape[2] == 2  # (x, y) coordinates


def test_motion_extractor_no_nans(motion_extractor, sample_image):
    out = motion_extractor(sample_image)
    assert not torch.isnan(out['kp_value']).any(), "Output contains NaNs"


def test_motion_extractor_gradients(motion_extractor, sample_image):
    out = motion_extractor(sample_image)
    out['kp_value'].sum().backward()

    for name, p in motion_extractor.named_parameters():
        assert p.grad is not None, f"{name} has no gradient!"


def test_motion_extractor_snapshot(motion_extractor, sample_image, data_regression):
    out = motion_extractor(sample_image)
    kp = out['kp_value'].detach().cpu().numpy()
    data_regression.check({"kp_value": kp.tolist()})
