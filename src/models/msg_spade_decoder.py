import torch
from torch import nn
import torch.nn.functional as F
from models.synthesis_decoder import SynthesisDecoder
from models.context_decoder import ContextDecoder


class MSGSpadeDecoder(nn.Module):
    def __init__(self,
                 context_decoder: ContextDecoder,
                 synthesis_decode: SynthesisDecoder):
        self.D_C = context_decoder
        self.D_S = synthesis_decode

    def forward(self, f_c, f_w, m_c):
        f_n = torch.cat([m_c, f_c], dim=1)

        fc = self.D_C.fc(f_c)
        fw = self.D_S.fc(f_w)

        # Looping through all middle resblocks from 0 to 5
        for i in range(0, 6):
            # Context decoder first
            x_s = getattr(self.D_C, f"G_middle_{i}").shortcut(fc, f_c)
            h_c = getattr(self.D_C, f"G_middle_{i}").norm_0(fc, f_c)

            # Synthesis decoder right after
            y_s = getattr(self.D_S, f"G_middle_{i}").shortcut(fw, f_w)
            h_w = getattr(self.D_S, f"G_middle_{i}").norm_0(fw, f_w)
            h_w = getattr(self.D_S, f"G_middle_{i}").gf_spade_1(f_n, h_c, h_w)
            h_w = getattr(self.D_S, f"G_middle_{i}").actvn(h_w)
            h_w = getattr(self.D_S, f"G_middle_{i}").conv_0(h_w)

            # Now context decoder can continues its loop
            h_c = getattr(self.D_C, f"G_middle_{i}").actvn(h_c)
            h_c = getattr(self.D_C, f"G_middle_{i}").conv_0(h_c)

            # Repeat for the second spade component
            h_c = getattr(self.D_C, f"G_middle_{i}").norm_1(h_c, f_c)

            h_w = getattr(self.D_S, f"G_middle_{i}").norm_1(h_w, f_w)
            h_w = getattr(self.D_S, f"G_middle_{i}").gf_spade_2(f_n, h_c, h_w)
            h_w = getattr(self.D_S, f"G_middle_{i}").actvn(h_w)
            h_w = getattr(self.D_S, f"G_middle_{i}").conv_1(h_w)

            h_c = getattr(self.D_C, f"G_middle_{i}").actvn(h_c)
            h_c = getattr(self.D_C, f"G_middle_{i}").conv_1(h_c)

            # Final re-sum
            out_c = h_c + x_s
            out_s = h_w + y_s

            # Prepping data for next resblock
            fc = out_c
            fw = out_s

        # Final touch for context decoder
        fc = self.D_C.up(fc)  # Bx512x64x64 -> Bx512x128x128
        fc = self.D_C.up_0(fc, f_c)  # Bx512x128x128 -> Bx256x128x128
        fc = self.D_C.up(fc)  # Bx256x128x128 -> Bx256x256x256
        fc = self.D_C.up_1(fc, f_c)  # Bx256x256x256 -> Bx64x256x256

        # Bx64x256x256 -> Bx3xHxW
        fc = self.D_C.conv_img(F.leaky_relu(f_c, 2e-1))
        fc = torch.sigmoid(fc)  # Bx3xHxW

        # Final touch for synthesis decoder
        fw = self.D_S.up(fw)  # Bx512x64x64 -> Bx512x128x128
        fw = self.D_S.up_0(fw, f_w)  # Bx512x128x128 -> Bx256x128x128
        fw = self.D_S.up(fw)  # Bx256x128x128 -> Bx256x256x256
        fw = self.D_S.up_1(fw, f_w)  # Bx256x256x256 -> Bx64x256x256

        # Bx64x256x256 -> Bx3xHxW
        fw = self.D_S.conv_img(F.leaky_relu(f_w, 2e-1))
        fw = torch.sigmoid(fw)  # Bx3xHxW

        return fw


if __name__ == "__main__":
    from loaders.loader import load_models
    D_C = load_models("D_C", pretrained=True)
    D_S = load_models("D_S", pretrained=True, strict=False)
    MSGSpadeDecoder(D_C, D_S)
