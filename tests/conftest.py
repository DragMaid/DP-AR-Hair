import pytest
from pathlib import Path


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
