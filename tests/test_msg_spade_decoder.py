import torch
import pytest
from models.msg_spade_decoder import MSGSpadeDecoder
from models.synthesis_decoder import SynthesisDecoder
from configs.model_config import model_config
from live_portrait.models.context_decoder import ContextDecoder


# @pytest.mark.skip
@pytest.mark.report_uss
@pytest.mark.report_tracemalloc
@pytest.mark.report_duration
def test_msg_spade_decoder_forward():
    # Create dummy decoders
    D_S = SynthesisDecoder(
        **model_config.synthesis_decoder_params.model_dump())
    D_C = ContextDecoder(
        **model_config.context_decoder_params.model_dump())

    model = MSGSpadeDecoder(D_C, D_S)

    # Prepare inputs: batch=2
    f_c = torch.randn(2, 256, 64, 64)
    f_w = torch.randn(2, 256, 64, 64)
    m_c = torch.randn(2, 1, 256, 256)  # Input Image will be 256x256

    out = model.forward(f_c, f_w, m_c)

    # Output should be a tensor with shape (B, 3, H, W) and values in [0,1]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 256, 256)
    assert torch.all(out >= 0) and torch.all(out <= 1)
