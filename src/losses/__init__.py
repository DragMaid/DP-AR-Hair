from .adversarial_loss import PatchGANDiscriminator
from .local_loss import HairLoss, FaceLoss
from .perceptual_loss import PerceptualLoss

__all__ = ["PatchGANDiscriminator", "HairLoss", "FaceLoss", "PerceptualLoss"]
