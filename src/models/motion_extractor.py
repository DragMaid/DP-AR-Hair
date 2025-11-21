from torch import nn
import torch

from models.convnextv2 import convnextv2_tiny
from models.util import filter_state_dict


class MotionExtractor(nn.Module):
    """
    Encoder network for the motion features (E_m) in the paper
    """

    def __init__(self, **kwargs):
        super(MotionExtractor, self).__init__()

        # Harcoded model is convnextv2_base
        self.detector = convnextv2_tiny(**kwargs)

    def load_pretrained(self, init_path: str):
        if init_path not in (None, ''):
            state_dict = torch.load(
                init_path, map_location=lambda storage, loc: storage)['model']
            state_dict = filter_state_dict(state_dict, remove_name='head')
            ret = self.detector.load_state_dict(state_dict, strict=False)
            print(f'Load pretrained model from {init_path}, ret: {ret}')

    def forward(self, x):
        out = self.detector(x)
        return out
