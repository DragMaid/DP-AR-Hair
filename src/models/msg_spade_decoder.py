import torch
from torch import nn
import torch.nn.functional as F
from models.synthesis_decoder import SynthesisDecoder
from live_portrait.models.context_decoder import ContextDecoder


class MSGSpadeDecoder(nn.Module):
    def __init__(self,
                 context_decoder: ContextDecoder,
                 synthesis_decode: SynthesisDecoder):
        super(MSGSpadeDecoder, self).__init__()
        self.D_C = context_decoder
        self.D_S = synthesis_decode

    def forward(self, f_c, f_w, m_c):
        # Resize m_c to be thesame as f_c first Bx2x64x64
        m_c_resized = F.interpolate(m_c, size=f_c.shape[-2:], mode='nearest')
        f_n = torch.cat([m_c_resized, f_c], dim=1)

        fc = self.D_C.fc(f_c)  # Bx256x64x64 -> Bx512x64x64
        fw = self.D_S.fc(f_w)  # Bx256x64x64 -> Bx512x64x64

        # Looping through all middle resblocks from 0 to 5
        for i in range(0, 6):
            fc, fw = self.resblock_forward(f"G_middle_{i}",
                                           f_c, fc, f_w, fw, f_n)

        # Transforming to valid RGB image
        for i in range(2):
            # Bx512x64x64 -> Bx512x128x128 -> Bx512x256x256
            fc = self.D_C.up(fc)
            # Bx512x64x64 -> Bx512x128x128 -> Bx512x256x256
            fw = self.D_S.up(fw)
            # Bx512x64x64 -> Bx512x128x128 -> Bx512x256x256
            f_n = self.D_S.up(f_n)
            # Bx512x64x64 -> Bx256x128x128 -> Bx64x256x256
            fc, fw = self.resblock_forward(f"up_{i}",
                                           f_c, fc, f_w, fw, f_n)

        # Bx64x256x256 -> Bx3xHxW
        fw = self.D_S.conv_img(F.leaky_relu(fw, 2e-1))
        fw = torch.sigmoid(fw)

        return fw

    def resblock_forward(self, name, f_c, fc, f_w, fw, f_n):
        x_s = getattr(self.D_C, name).shortcut(fc, f_c)
        h_c = getattr(self.D_C, name).norm_0(fc, f_c)

        # Synthesis decoder right after
        y_s = getattr(self.D_S, name).shortcut(fw, f_w)
        h_w = getattr(self.D_S, name).norm_0(fw, f_w)
        h_w = getattr(self.D_S, name).gf_spade_1(f_n, h_c, h_w)
        h_w = getattr(self.D_S, name).actvn(h_w)
        h_w = getattr(self.D_S, name).conv_0(h_w)

        # Now context decoder can continues its loop
        h_c = getattr(self.D_C, name).actvn(h_c)
        h_c = getattr(self.D_C, name).conv_0(h_c)

        # Repeat for the second spade component
        h_c = getattr(self.D_C, name).norm_1(h_c, f_c)

        h_w = getattr(self.D_S, name).norm_1(h_w, f_w)
        h_w = getattr(self.D_S, name).gf_spade_2(f_n, h_c, h_w)
        h_w = getattr(self.D_S, name).actvn(h_w)
        h_w = getattr(self.D_S, name).conv_1(h_w)

        h_c = getattr(self.D_C, name).actvn(h_c)
        h_c = getattr(self.D_C, name).conv_1(h_c)

        # Final re-sum
        out_c = h_c + x_s
        out_s = h_w + y_s

        return out_c, out_s
