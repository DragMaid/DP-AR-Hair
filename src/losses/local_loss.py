import torch
import torch.nn as nn


class HairLoss(nn.Module):
    def __init__(self):
        super(HairLoss, self).__init__()

    def forward(self, m_h, I_p, I_d):
        return torch.norm((m_h * (I_d - I_p)), p=1)


class FaceLoss(nn.Module):
    def __init__(self):
        super(FaceLoss, self).__init__()

    def forward(self, m_f, I_p, I_d):
        return torch.norm((m_f * (I_d - I_p)), p=1)
