import torch
import torch.nn as nn
from loaders.loader import load_models, ModelRegistry
from loaders.downloader import download_weights
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial_loss import PatchGANDiscriminator, weights_init
from losses.local_loss import HairLoss, FaceLoss
from models.msg_spade_decoder import MSGSpadeDecoder
from face_parsing.models.utils import get_mask_by_idx
from configs.pipeline_config import pipeline_config as pco
from torchvision import transforms


class TrainingPipeline:
    def __init__(self):
        self.E_H = load_models("E_H", pretrained=True, freeze=True)
        self.E_M = load_models("E_M", pretrained=True, freeze=True)
        self.E_C = load_models("E_C", pretrained=False)  # E_C is trainable
        self.W = load_models("W", pretrained=True, freeze=True)
        self.M_C = load_models("M_C", pretrained=True, freeze=True)

        # D_S will load and freeze all params except for GF_SPADEs
        self.D_S = load_models("D_S", pretrained=True,
                               strict=False, freeze=True)

        self.D_C = load_models("D_C", pretrained=True, freeze=True)

        self.D = MSGSpadeDecoder(self.D_C, self.D_S)

        # TODO: write a util function that init for different IIHTs
        # This is specifically for loading HairFastGAN IIHT
        IIHT_NAME = "IIHT1"
        record = ModelRegistry.get_registry(IIHT_NAME)
        w_options = record["weight"]["options"]
        dest = w_options["local_dir"] / \
            w_options["allow_patterns"][0].split("/")[0]
        if not dest.exists():
            download_weights(record["weight"]["type"], w_options)
        self.IIHT = load_models(IIHT_NAME, pretrained=False)

        # This is for loss functions
        # Normalization for VGG16
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.L_p = PerceptualLoss()

        self.disc_criterion = nn.BCEWithLogitsLoss()
        self.L_adv = PatchGANDiscriminator(n_in_channels=3)  # RGB
        self.L_adv.apply(weights_init)

        self.L_hair = HairLoss()
        self.L_face = FaceLoss()

        self.L_global = nn.L1Loss()

        # Optimizer declarations
        self.generator_trainable_params = []
        self.generator_trainable_params += filter(
            lambda p: p.requires_grad, self.D_S.parameters())
        self.generator_trainable_params += self.E_C.parameters()

        self.generator_optimizer = torch.optim.Adam(
            self.generator_trainable_params, lr=pco.training.learn_rate)

        self.disc_optimizer = torch.optim.Adam(
            self.L_adv.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def forward(self, I_d, I_s, R):
        self.generator_optimizer.zero_grad()
        I_d_dilde = self.IIHT(I_d, R)

        f_c = self.E_C(I_d_dilde)

        f_h = self.E_H(I_s)
        f_m = self.E_M(I_s)
        f_w = self.W(f_h, f_m)

        m_c = get_mask_by_idx(I_d_dilde, self.M_C)

        I_p = self.D(f_c, f_w, m_c)

        p_loss = self.L_p(self.normalize(I_p), self.normalize(I_d))

        self.disc_optimizer.zero_grad()

        pred_real = self.L_adv(I_d)
        target_real = torch.ones_like(pred_real)
        loss_real = self.disc_criterion(pred_real, target_real)

        pred_fake = self.L_adv(I_p)
        target_fake = torch.zeros_like(pred_fake)
        loss_fake = self.disc_criterion(pred_fake, target_fake)

        a_loss = (loss_fake + loss_real) / 2
        a_loss.backward()
        self.disc_optimizer.step()

        h_loss = self.L_hair(I_d, I_p)
        f_loss = self.L_face(I_d, I_p)
        g_loss = self.L_global(I_d, I_p)

        t = pco.training
        total_loss = t.p_rate * p_loss + t.adv_rate * \
            a_loss + t.h_rate * h_loss + t.f_rate * f_loss + t.rec_rate * g_loss
        total_loss.backward()

        self.generator_optimizer.step()
