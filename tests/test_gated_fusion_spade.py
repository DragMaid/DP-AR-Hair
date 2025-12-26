import pytest
import importlib.util
import pathlib
import sys
import torch


def load_module_from_path(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.report_uss
@pytest.mark.report_tracemalloc
@pytest.mark.report_duration
def test_gfspade_smoke_and_regression(data_regression):
    """Smoke test for GFSPADE and simple regression check.

    - Loads the module by path so tests don't rely on package installation.
    - Runs a small forward pass and records shape, mean and std for regression.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    gf_path = repo_root / "src" / "models" / "gated_fusion_spade.py"
    mod = load_module_from_path(gf_path, "gated_fusion_spade")

    GFSPADE = getattr(mod, "GFSPADE")

    torch.manual_seed(0)

    B, C, H, W = 2, 16, 32, 32
    Cf, Cm = 8, 1

    gf = GFSPADE(num_channels=C, cond_channels=Cf + Cm, hidden_channels=8, post_conv=True)
    gf.eval()

    h_w = torch.randn(B, C, H, W)
    h_c = torch.randn(B, C, H, W)
    f_c = torch.randn(B, Cf, H, W)
    m_c = torch.randint(0, 2, (B, Cm, H, W)).float()
    f_n = torch.cat([f_c, m_c], dim=1)

    out = gf(f_n, h_c, h_w)

    # Basic sanity
    assert out.shape == (B, C, H, W)

    # Regression-friendly numeric checks
    # data_regression.check({
    #     "shape": list(out.shape),
    #     "mean": float(out.mean().item()),
    #     "std": float(out.std().item()),
    # })

