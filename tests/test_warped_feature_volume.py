import torch
import pytest
from configs.model_config import model_config
from live_portrait.models.appearance_feature_extractor import AppearanceFeatureExtractor
from live_portrait.models.motion_extractor import MotionExtractor
from live_portrait.models.warping_network import WarpingNetwork
from live_portrait.models.context_decoder import ContextDecoder


@pytest.fixture
def appearance_extractor():
    return AppearanceFeatureExtractor(
        **model_config.appearance_feature_extractor_params.model_dump())


@pytest.fixture
def motion_extractor():
    return MotionExtractor(
        **model_config.motion_extractor_params.model_dump())


@pytest.fixture
def warping_network():
    return WarpingNetwork(
        **model_config.warping_module_params.model_dump())


@pytest.fixture
def context_decoder():
    return ContextDecoder(
        **model_config.context_decoder_params.model_dump())


@pytest.fixture
def sample_image():
    return torch.randn(2, 3, 256, 256)  # Input image size


def test_forward_flow(context_decoder, appearance_extractor, motion_extractor, warping_network, sample_image):
    f_h = appearance_extractor(sample_image)
    f_m = motion_extractor(sample_image)["kp"].view(
        sample_image.size(0), -1, 3)
    f_m_d = motion_extractor(sample_image)["kp"].view(
        sample_image.size(0), -1, 3)
    f_w = warping_network(feature_3d=f_h, kp_source=f_m, kp_driving=f_m_d)
    out = context_decoder(f_w["out"])
    assert out.size() == torch.Size([2, 3, 512, 512])
