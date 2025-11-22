import pytest
from pathlib import Path
import numpy as np
import torch
import random

# Enable deterministic algorithms globally for reproducible tests
torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session", autouse=True)
def seed_all():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    parser.addini("original_datadir",
                  "my own original_datadir for pytest-regressions")
