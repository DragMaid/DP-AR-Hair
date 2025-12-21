import torch
import torch.nn as nn
from torchvision import transforms
from losses.perceptual_loss import PerceptualLoss
from losses.local_loss import HairLoss, FaceLoss
from configs.pipeline_config import pipeline_config as pco


class LossHandler:
    """
    Stateless loss computation. All losses are instances owned by this class.
    """

    def __init__(self, device):
        self.device = device

        # Normalization for perceptual loss
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Loss modules
        self.L_p = PerceptualLoss().to(device)
        self.L_hair = HairLoss().to(device)
        self.L_face = FaceLoss().to(device)
        self.L_global = nn.L1Loss().to(device)

        # Discriminator criterion
        # Use BCEWithLogitsLoss (wrapper) for stability
        self.disc_criterion = nn.BCEWithLogitsLoss().to(device)

        # Loss weights from config
        self.weights = pco.training.loss

    @torch.cuda.amp.autocast()
    def compute_generator_losses(self, I_d, I_p, m_c, m_f, discriminator):
        """
        Compute all generator losses.

        Args:
            I_d: destination image (B, C, H, W)
            I_p: prediction (B, C, H, W)
            m_c: Hair mask (B, 1, H, W)
            m_f: Non hair mask (B, 1, H, W)
            discriminator: discriminator network (for adversarial loss)

        Returns:
            dict of losses:
                total_loss: weighted sum
                p_loss: perceptual loss
                h_loss: hair loss
                f_loss: face loss
                g_loss: global L1 loss
                a_gen_loss: adversarial loss for generator
        """
        # Perceptual loss
        p_loss = self.L_p(self.normalize(I_p), self.normalize(I_d))

        # Local losses
        h_loss = self.L_hair(m_c, I_d, I_p)
        f_loss = self.L_face(m_f, I_d, I_p)

        # Global reconstruction
        g_loss = self.L_global(I_d, I_p)

        # Adversarial loss (generator tries to fool discriminator)
        pred_fake = discriminator(I_p)
        target_real = torch.ones_like(pred_fake)
        a_gen_loss = self.disc_criterion(pred_fake, target_real)

        # Weighted sum
        total_loss = (
            self.weights.p_rate * p_loss +
            self.weights.adv_rate * a_gen_loss +
            self.weights.h_rate * h_loss +
            self.weights.f_rate * f_loss +
            self.weights.rec_rate * g_loss
        )

        return {
            "total_loss": total_loss,
            "p_loss": p_loss,
            "h_loss": h_loss,
            "f_loss": f_loss,
            "g_loss": g_loss,
            "a_gen_loss": a_gen_loss,
        }

    def compute_discriminator_loss(self, I_d, I_p, discriminator):
        """
        Compute discriminator loss (real vs fake).

        Args:
            I_d: real image (B, C, H, W)
            I_p: fake image (B, C, H, W) - must be detached
            discriminator: discriminator network

        Returns:
            disc_loss: scalar tensor
        """
        # Real images should be classified as 1
        pred_real = discriminator(I_d)
        target_real = torch.ones_like(pred_real)
        loss_real = self.disc_criterion(pred_real, target_real)

        # Fake images should be classified as 0
        pred_fake = discriminator(I_p.detach())
        target_fake = torch.zeros_like(pred_fake)
        loss_fake = self.disc_criterion(pred_fake, target_fake)

        disc_loss = (loss_real + loss_fake) / 2
        return disc_loss
