import torch
import pytest
from loaders.loader import load_models
from face_parsing.models.utils import get_mask_by_idx


@pytest.fixture
def parser():
    return load_models("M_C", pretrained=True, freeze=True)


@pytest.fixture
def sample_single_image():
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_batch_image():
    return torch.randn(5, 3, 256, 256)


def test_single_inference(parser, sample_single_image):
    mask = get_mask_by_idx(sample_single_image, parser)
    assert mask.size() == (1, 1, 256, 256)


def test_batch_inference(parser, sample_batch_image):
    mask = get_mask_by_idx(sample_batch_image, parser)
    assert mask.size() == (5, 1, 256, 256)
