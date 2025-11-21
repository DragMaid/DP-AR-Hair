import torch
import pytest
from configs.configs import config
from models.warping_network import WarpingNetwork


@pytest.fixture
def warping_network():
    torch.manual_seed(0)
    params = config.warping_module_params.model_dump()
    dense = params.pop("dense_motion_params")

    return WarpingNetwork(
        **params,
        dense_motion_params=dense
    )


@pytest.fixture
def sample_data():
    torch.manual_seed(0)
    feature_3d = torch.randn(2, 32, 16, 64, 64)
    kp_driving = torch.randn(2, 21, 3)
    kp_source = torch.randn(2, 21, 3)
    return feature_3d, kp_driving, kp_source


# -----------------------------------------------------------
# Structural Tests
# -----------------------------------------------------------

def test_output_keys(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"out", "deformation", "occlusion_map"}


def test_output_shapes(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    # out: B x 256 x 64 x 64
    assert out["out"].shape == (2, 256, 64, 64)

    # deformation: B x 16 x 64 x 64 x 3
    assert out["deformation"].shape == (2, 16, 64, 64, 3)

    # occlusion_map: B x 1 x 64 x 64 OR None
    occ = out["occlusion_map"]
    if occ is not None:
        assert occ.shape == (2, 1, 64, 64)


def test_no_nans(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    assert not torch.isnan(out["out"]).any()
    assert not torch.isnan(out["deformation"]).any()
    if out["occlusion_map"] is not None:
        assert not torch.isnan(out["occlusion_map"]).any()


# -----------------------------------------------------------
# Gradient Test
# -----------------------------------------------------------

def test_gradients_flow(warping_network, sample_data):
    feature_3d, kp_driving, kp_source = sample_data

    out = warping_network(feature_3d, kp_driving, kp_source)
    loss = out["out"].sum()
    loss.backward()

    for name, p in warping_network.named_parameters():
        assert p.grad is not None, f"Gradient missing for {name}"


# -----------------------------------------------------------
# Snapshot Test
# -----------------------------------------------------------

def test_snapshot_regression(warping_network, sample_data, data_regression):
    feature_3d, kp_driving, kp_source = sample_data
    out = warping_network(feature_3d, kp_driving, kp_source)

    snapshot = {
        "out_mean": float(out["out"].mean().item()),
        "out_std": float(out["out"].std().item()),
        "deformation_mean": float(out["deformation"].mean().item()),
        "deformation_std": float(out["deformation"].std().item()),
        "use_occlusion": out["occlusion_map"] is not None,
    }

    data_regression.check(snapshot)
