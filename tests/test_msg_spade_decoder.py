import torch
from models.msg_spade_decoder import MSGSpadeDecoder


class DummyBlock:
    def shortcut(self, x, ref):
        return x

    def norm_0(self, x, ref):
        return x

    def norm_1(self, x, ref):
        return x

    def gf_spade_1(self, f_n, h_c, h_w):
        return h_w

    def gf_spade_2(self, f_n, h_c, h_w):
        return h_w

    def actvn(self, x):
        return x

    def conv_0(self, x):
        return x

    def conv_1(self, x):
        return x


class DummyContextDecoder:
    def __init__(self):
        # create 6 middle blocks
        for i in range(6):
            setattr(self, f"G_middle_{i}", DummyBlock())

    def fc(self, x):
        return x

    def up(self, x):
        return x

    def up_0(self, x, ref):
        return x

    def up_1(self, x, ref):
        return x

    def conv_img(self, x):
        # Expect x to have at least 3 channels; return first 3 channels
        return x[:, :3, :, :]


class DummySynthesisDecoder(DummyContextDecoder):
    pass


def test_msg_spade_decoder_forward():
    # Create dummy decoders
    D_C = DummyContextDecoder()
    D_S = DummySynthesisDecoder()

    model = MSGSpadeDecoder(D_C, D_S)

    # Prepare inputs: batch=2, channels>=3, H=W=8
    B, C, H, W = 2, 8, 8, 8
    f_c = torch.randn(B, C, H, W)
    f_w = torch.randn(B, C, H, W)
    m_c = torch.randn(B, C, H, W)

    out = model.forward(f_c, f_w, m_c)

    # Output should be a tensor with shape (B, 3, H, W) and values in [0,1]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, 3, H, W)
    assert torch.all(out >= 0) and torch.all(out <= 1)

