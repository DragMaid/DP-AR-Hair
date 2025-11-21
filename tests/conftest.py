import pytest
from pathlib import Path
import numpy as np
import torch

# Enable deterministic algorithms globally for reproducible tests
torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def original_datadir(request) -> Path:
    config = request.config
    return config.rootpath / Path(config.getini("original_datadir"))


@pytest.fixture(scope="session")
def lazy_datadir(request) -> Path:
    config = request.config
    return config.rootpath / Path(config.getini("lazy_datadir"))


def pytest_addoption(parser):
    parser.addini("lazy_datadir", "my own datadir for pytest-regressions")
    parser.addini("original_datadir", "my own original_datadir for pytest-regressions")


# Approximate comparison helper for snapshot checks
def _approx_data(out, expected, tol=1e-5):
    out = np.array(out)
    expected = np.array(expected)
    assert np.allclose(out, expected, atol=tol)


@pytest.fixture
def approx_data():
    """
    Fixture that returns a callable approx_data(out, expected, tol=1e-5)
    for use in tests (e.g. approx_data(out.tolist(), snapshot_data)).
    """
    return _approx_data
