import torch
import pytest
from configs.configs import config
from models.motion_extractor import MotionExtractor


@pytest.fixture
def motion_extractor():
    return MotionExtractor(
        **config.motion_extractor_params.model_dump())


@pytest.fixture
def sample_image():
    return torch.randn(2, 3, 256, 256)


def test_motion_extractor_output_structure(motion_extractor, sample_image):
    out = motion_extractor(sample_image)
    assert isinstance(out, dict)
    assert 'kp' in out
    assert isinstance(out['kp'], torch.Tensor)


def test_motion_extractor_kp_shape(motion_extractor, sample_image):
    out = motion_extractor(sample_image)
    kp = out['kp']
    assert kp.ndim == 2
    assert kp.shape[0] == 2  # batch size
    assert kp.shape[1] == 3 * config.motion_extractor_params.num_kp


def test_motion_extractor_no_nans(motion_extractor, sample_image):
    out = motion_extractor(sample_image)
    assert not torch.isnan(out['kp']).any(), "Output contains NaNs"


def test_motion_extractor_gradients(motion_extractor, sample_image):
    out = motion_extractor(sample_image)
    out['kp'].sum().backward()

# Check only parameters that could affect kp
    for name, p in motion_extractor.named_parameters():
        if 'fc_kp' in name or 'stages' in name or 'downsample_layers' in name:
            assert p.grad is not None, f"{name} not used for kp!"


def test_motion_extractor_snapshot(motion_extractor, sample_image, data_regression):
    out = motion_extractor(sample_image)
    kp = out['kp'].detach().cpu().numpy()
    data_regression.check({"kp": kp.tolist()},
                          default_tolerance={"atol": 1e-5})
