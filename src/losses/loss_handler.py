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
        self.L_global = nn.L1Loss(reduction='mean').to(device)

        # Discriminator criterion
        # Use BCEWithLogitsLoss (wrapper) for stability
        self.disc_criterion = nn.BCEWithLogitsLoss().to(device)

        # Loss weights from config
        self.weights = pco.training.loss

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
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            # Perceptual loss
            perceptual_loss = self.weights.p_rate * self.L_p(self.normalize(
                I_p), self.normalize(I_d))

            # Local losses
            hair_loss = self.weights.h_rate * self.L_hair(m_c, I_d, I_p)
            face_loss = self.weights.f_rate * self.L_face(m_f, I_d, I_p)

            # Global reconstruction
            global_loss = self.weights.rec_rate * self.L_global(I_d, I_p)

        # Generator just wants to fool the discriminator so no real_loss
        # Adversarial loss (generator tries to fool discriminator)
        # Make sure its in FP32
        pred_fake = discriminator(I_p)
        target_real = torch.ones_like(pred_fake)
        a_gen_loss = self.weights.adv_rate * \
            self.disc_criterion(pred_fake, target_real)

        # Calculate the gradient contribution of each loss

        # Weighted sum
        generator_loss = perceptual_loss + a_gen_loss + \
            hair_loss + face_loss + global_loss

        return {
            "generator_loss": generator_loss,
            "perceptual_loss": perceptual_loss,
            "hair_loss": hair_loss,
            "face_loss": face_loss,
            "global_loss": global_loss,
            "adversarial_gen_loss": a_gen_loss,
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
        # Real images should be classified as 0.9 (label smoothing)
        pred_real = discriminator(I_d)
        target_real = torch.ones_like(pred_real) * 0.9
        loss_real = self.disc_criterion(pred_real, target_real)

        # Make sure that I_p is detached first
        # Fake images should be classified as 0.1 (label smoothing)
        pred_fake = discriminator(I_p)
        target_fake = torch.zeros_like(pred_fake) + 0.1
        loss_fake = self.disc_criterion(pred_fake, target_fake)

        disc_loss = (loss_real + loss_fake) / 2
        return disc_loss
