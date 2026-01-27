import torch
import torch.nn as nn


class HairLoss(nn.Module):
    def __init__(self):
        super(HairLoss, self).__init__()

    def forward(self, m_h, I_p, I_d):
        # Normalize by number of hair pixels
        return torch.sum(torch.abs(m_h * (I_d - I_p))) / (m_h.sum() + 1e-6)


class FaceLoss(nn.Module):
    def __init__(self):
        super(FaceLoss, self).__init__()

    def forward(self, m_f, I_p, I_d):
        # Normalize by number of face pixels
        return torch.sum(torch.abs(m_f * (I_d - I_p))) / (m_f.sum() + 1e-6)
