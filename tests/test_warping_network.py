import torch
import pytest
from src.models.warping_network import WarpingNetwork


@pytest.fixture
def warping_network():
    torch.manual_seed(0)
    return WarpingNetwork(
        num_kp=10,
        block_expansion=64,
        max_features=1024,
        num_down_blocks=2,
        reshape_channel=32,
        estimate_occlusion_map=True,
        dense_motion_params={
            'block_expansion': 64,
            'max_features': 1024,
            'num_blocks': 4,
            'pad': 'reflect',
            'use_mask': False
        }
    )


@pytest.fixture
def sample_data():
    torch.manual_seed(0)
    feature_3d = torch.randn(2, 32, 16, 64, 64)
    kp_driving = torch.randn(2, 10, 2)
    kp_source = torch.randn(2, 10, 2)
    return feature_3d, kp_driving, kp_source


def test_warping_network_output_structure(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    assert isinstance(out, dict)
    assert 'out' in out
    assert 'deformation' in out
    assert 'occlusion_map' in out


def test_warping_network_output_shape(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    assert out['out'].ndim == 4
    assert out['out'].shape[0] == 2  # batch size


def test_warping_network_deformation_shape(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    assert out['deformation'].ndim == 5
    assert out['deformation'].shape[-1] == 3  # spatial coordinates


def test_warping_network_no_nans(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    assert not torch.isnan(out['out']).any(), "Output contains NaNs"
    assert not torch.isnan(out['deformation']).any(), "Deformation contains NaNs"


def test_warping_network_gradients(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)
    out['out'].sum().backward()

    for name, p in warping_network.named_parameters():
        assert p.grad is not None, f"{name} has no gradient!"


def test_warping_network_snapshot(warping_network, sample_data, data_regression):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    snapshot_data = {
        "output": out['out'].detach().cpu().numpy().tolist(),
        "deformation_shape": list(out['deformation'].shape),
    }
    data_regression.check(snapshot_data)
