from loaders.loader import load_models
from models.msg_spade_decoder import MSGSpadeDecoder
from pipelines.hair_mask import get_hair_mask


class Pipeline:
    def __init__(self):
        self.E_H = load_models("E_H", pretrained=True)
        self.E_M = load_models("E_H", pretrained=True)
        self.E_C = load_models("E_H", pretrained=False)
        self.W = load_models("W", pretrained=True)

        self.D_S = load_models("D_S", pretrained=True, strict=False)
        self.D_C = load_models("D_C", pretrained=True)

        self.D = MSGSpadeDecoder(self.D_C, self.D_S)
        self.IIHT = StableHair()  # TODO: actually add the model later

    def forward(self, I_d, I_s, R):
        I_d_dilde = self.IIHT(I_d, R)
        f_c = self.E_C(I_d_dilde)

        f_h = self.E_H(I_s)
        f_m = self.E_M(I_s)
        f_w = self.W(f_h, f_m)

        m_c = get_hair_mask(I_d_dilde, model)

        I_p = self.D(f_c, f_w, m_c)
