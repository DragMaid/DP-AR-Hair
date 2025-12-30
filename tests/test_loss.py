import pytest
import torch
from losses.loss_handler import LossHandler
from losses.adversarial_loss import PatchGANDiscriminator

size = 256


@pytest.fixture
def I_d():
    return torch.randn([5, 3, size, size])


@pytest.fixture
def I_p():
    return torch.randn([5, 3, size, size])


@pytest.fixture
def m_c():
    return torch.randn([5, 1, size, size])


@pytest.fixture
def m_f():
    return torch.randn([5, 1, size, size])


@pytest.fixture
def discriminator():
    return PatchGANDiscriminator(n_in_channels=3)


@pytest.fixture
def losses():
    return LossHandler(torch.device("cpu"))


@pytest.mark.report_uss
@pytest.mark.report_tracemalloc
@pytest.mark.report_duration
def test_generator_losses(I_d, I_p, m_c, m_f, discriminator, losses):
    loss = losses.compute_generator_losses(I_d, I_p, m_c, m_f, discriminator)
    print(loss)
    assert loss is not None


@pytest.mark.report_uss
@pytest.mark.report_tracemalloc
@pytest.mark.report_duration
def test_disc_loss(I_d, I_p, discriminator, losses):
    loss = losses.compute_discriminator_loss(I_d, I_p, discriminator)
    print(loss)
    assert loss is not None
