import torch
from loaders.loader import load_models
from models.msg_spade_decoder import MSGSpadeDecoder
from configs.model_config import model_config


class InferencePipeline:
    """
    InferencePipeline:
    - provides inference()
    - provides load_checkpoint()
    """

    def __init__(self, device, loaded=True):
        self.device = device

        # --- Generator (G) ---
        self.E_H = load_models("E_H", pretrained=loaded,
                               freeze=True).to(self.device)
        self.E_M = load_models("E_M", pretrained=loaded,
                               freeze=True).to(self.device)
        self.E_C = load_models("E_C", pretrained=False).to(
            self.device)  # trainable
        self.W = load_models("W", pretrained=loaded,
                             freeze=True).to(self.device)
        self.M_C = load_models("M_C", pretrained=True,
                               freeze=True).to(self.device)

        # WARN: This is not very configurable
        # Setting the upscale to 2 so output is 512x512
        D_S_config = model_config.synthesis_decoder_params.model_dump()
        D_S_config["upscale"] = 2

        # D_S will load and freeze all params except for GF_SPADEs
        self.D_S = load_models("D_S", pretrained=loaded,
                               strict=False, freeze=True,
                               params=D_S_config).to(self.device)
        self.D_C = load_models("D_C", pretrained=loaded,
                               freeze=True).to(self.device)
        self.D = MSGSpadeDecoder(self.D_C, self.D_S)

        self.modules_to_load = {
            "E_C": self.E_C,
            "D_S": self.D_S,
        }

    @torch.no_grad()
    def inference(self, I_s, I_d_t):
        # TODO: should I send the image right back or should I make it into an entire video ?
        # For the pipeline let's just send the image back
        I_s = I_s.to(self.device)
        I_d_t = I_d_t.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            f_c = self.E_C(I_d_t)
            f_h = self.E_H(I_s)

            f_m = self.E_M(I_s)["kp"].view(I_s.size(0), -1, 3)
            f_m_d = self.E_M(I_d_t)["kp"].view(I_s.size(0), -1, 3)

            f_w = self.W(feature_3d=f_h, kp_source=f_m,
                         kp_driving=f_m_d)['out']

            # Mask is not strictly required in inference
            # Keep batch and channel, zeros the rest
            m_c = torch.zeros_like(f_c[:, :1])
            I_p = self.D(f_c, f_w, m_c)

            return I_p

    def load_checkpoint(self, path: str):
        """
        Load checkpoint into respective modules. Returns checkpoint dict.
        """
        ck = torch.load(path, map_location=self.device)
        for name, module in self.modules_to_load.items():
            if name in ck:
                try:
                    module.load_state_dict(ck[name], strict=False)
                except Exception:
                    module.load_state_dict(ck[name])
        return ck
