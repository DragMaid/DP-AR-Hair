# training/pipeline_ext.py
import torch
import torch.nn as nn
from torchvision import transforms
from losses.perceptual_loss import PerceptualLoss
from losses.adversarial_loss import PatchGANDiscriminator, weights_init
from losses.local_loss import HairLoss, FaceLoss
from face_parsing.models.utils import get_mask_by_idx
from configs.pipeline_config import pipeline_config as pco
from loaders.loader import load_models, ModelRegistry
from loaders.downloader import download_weights
from models.msg_spade_decoder import MSGSpadeDecoder
import os


class TrainingPipeline:
    """
    Extended TrainingPipeline:
    - provides train_step(batch, device, scaler=None, reference_mode='self')
    - provides save_checkpoint / load_checkpoint
    - collects a minimal set of modules to save
    """

    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device

        # --- load models (kept same as your snippet) ---
        self.E_H = load_models("E_H", pretrained=True,
                               freeze=True).to(self.device)
        self.E_M = load_models("E_M", pretrained=True,
                               freeze=True).to(self.device)
        self.E_C = load_models("E_C", pretrained=False).to(
            self.device)  # trainable
        self.W = load_models("W", pretrained=True, freeze=True).to(self.device)
        self.M_C = load_models("M_C", pretrained=True,
                               freeze=True).to(self.device)

        # D_S will load and freeze all params except for GF_SPADEs
        self.D_S = load_models("D_S", pretrained=True,
                               strict=False, freeze=True).to(self.device)
        self.D_C = load_models("D_C", pretrained=True,
                               freeze=True).to(self.device)
        self.D = MSGSpadeDecoder(self.D_C, self.D_S)

        # IIHT loading (keeps your logic)
        IIHT_NAME = "IIHT1"
        record = ModelRegistry.get_registry(IIHT_NAME)
        w_options = record["weight"]["options"]
        dest = w_options["local_dir"] / \
            w_options["allow_patterns"][0].split("/")[0]
        if not dest.exists():
            download_weights(record["weight"]["type"], w_options)
        self.IIHT = load_models(IIHT_NAME, pretrained=False)

        # Losses and normalizer
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.L_p = PerceptualLoss().to(self.device)

        # Use BCEWithLogitsLoss (wrapper) for stability
        self.disc_criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.L_adv = PatchGANDiscriminator(n_in_channels=3).to(self.device)
        self.L_adv.apply(weights_init)

        self.L_hair = HairLoss().to(self.device)
        self.L_face = FaceLoss().to(self.device)
        self.L_global = nn.L1Loss().to(self.device)

        # Optimizer declarations (mirrors your original selection)
        self.generator_trainable_params = []
        # include any parameters that require grad from D_S and E_C
        self.generator_trainable_params += [
            p for p in self.D_S.parameters() if p.requires_grad]
        self.generator_trainable_params += [
            p for p in self.E_C.parameters() if p.requires_grad]

        self.generator_optimizer = torch.optim.Adam(
            self.generator_trainable_params,
            lr=pco.training.generator.learn_rate,
            betas=pco.training.generator.betas)

        self.disc_optimizer = torch.optim.Adam(
            self.L_adv.parameters(),
            lr=pco.training.discriminator.learn_rate,
            betas=pco.training.discriminator.betas)

        # Modules we want to save when checkpointing (minimal set)
        self.modules_to_save = {
            "E_C": self.E_C,
            "D_S": self.D_S,
            "L_adv": self.L_adv
        }

    @torch.cuda.amp.autocast()
    def _compute_losses(self, I_d, I_s, I_p, I_d_dilde):
        """Compute all losses (returns scalar tensors)"""
        p_loss = self.L_p(self.normalize(I_p), self.normalize(I_d))
        pred_real = self.L_adv(I_d)
        target_real = torch.ones_like(pred_real)
        loss_real = self.disc_criterion(pred_real, target_real)

        pred_fake = self.L_adv(I_p.detach())
        target_fake = torch.zeros_like(pred_fake)
        loss_fake = self.disc_criterion(pred_fake, target_fake)
        a_loss = (loss_fake + loss_real) / 2

        h_loss = self.L_hair(I_d, I_p)
        f_loss = self.L_face(I_d, I_p)
        g_loss = self.L_global(I_d, I_p)

        pred_fake_G = self.L_adv(I_p)
        target_fake_G = torch.zeros_like(pred_fake_G)
        a_fake_loss = self.disc_criterion(pred_fake_G, target_fake_G)

        t = pco.training.loss
        total_loss = t.p_rate * p_loss + t.adv_rate * a_fake_loss + \
            t.h_rate * h_loss + t.f_rate * f_loss + t.rec_rate * g_loss

        losses = {
            "p_loss": p_loss,
            "a_loss_disc": a_loss,
            "a_fake_loss_gen": a_fake_loss,
            "h_loss": h_loss,
            "f_loss": f_loss,
            "g_loss": g_loss,
            "total_loss": total_loss
        }
        return losses

    def train_step(self, I_s, I_d, I_r, scaler=None):
        """
        Perform a single training step (one batch).
        - I_s, I_d, I_r: tensors (BCHW) on CPU or device
        - scaler: optional torch.cuda.amp.GradScaler for mixed precision
        Returns dict of scalars (floats) for logging.
        """

        ALIGNMENT_MODE = "Auto"  # Auto, On, Off
        need_alignment = any(img.size != (1024, 1024) for img in (I_r, I_d))
        perform_align = ALIGNMENT_MODE == "On" or (
            ALIGNMENT_MODE == "Auto" and need_alignment)

        if perform_align:
            I_d_dilde, _, _, _ = self.IIHT(I_d, I_r, I_d, align=True)
        else:
            I_d_dilde = self.IIHT(I_d, I_r, I_d)

        f_c = self.E_C(I_d_dilde)
        f_h = self.E_H(I_s)
        f_m = self.E_M(I_s)
        f_w = self.W(f_h, f_m)
        m_c = get_mask_by_idx(I_d_dilde, self.M_C)
        I_p = self.D(f_c, f_w, m_c)

        # --- Discriminator update ---
        # zero grad
        self.disc_optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            pred_real = self.L_adv(I_d)
            target_real = torch.ones_like(pred_real)
            loss_real = self.disc_criterion(pred_real, target_real)

            pred_fake = self.L_adv(I_p.detach())
            target_fake = torch.zeros_like(pred_fake)
            loss_fake = self.disc_criterion(pred_fake, target_fake)
            disc_loss = (loss_real + loss_fake) / 2

        # backprop discriminator
        if scaler is not None:
            scaler.scale(disc_loss).backward()
            scaler.step(self.disc_optimizer)
        else:
            disc_loss.backward()
            self.disc_optimizer.step()

        # --- Generator update ---
        self.generator_optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            losses = self._compute_losses(I_d, I_s, I_p, I_d_dilde)
        gen_loss = losses["total_loss"]

        if scaler is not None:
            scaler.scale(gen_loss).backward()
            scaler.step(self.generator_optimizer)
            scaler.update()
        else:
            gen_loss.backward()
            self.generator_optimizer.step()

        # return python scalars for logging
        logs = {k: float(v.detach().cpu()) for k, v in losses.items()}
        logs["disc_loss"] = float(disc_loss.detach().cpu())
        return logs

    def save_checkpoint(self, path: str, epoch: int, extra: dict = None):
        """
        Save the minimal checkpoint (selected modules + optimizers).
        extra: optional dict with arbitrary metadata.
        """
        payload = {"epoch": epoch}
        # modules
        for name, module in self.modules_to_save.items():
            try:
                payload[name] = module.state_dict()
            except Exception:
                # fallback: skip
                pass

        # optimizers
        payload["generator_optimizer"] = self.generator_optimizer.state_dict()
        payload["disc_optimizer"] = self.disc_optimizer.state_dict()
        if extra:
            payload["extra"] = extra

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(self, path: str, load_optimizers: bool = True):
        """
        Load checkpoint into respective modules. Returns checkpoint dict.
        """
        ck = torch.load(path, map_location=self.device)
        for name, module in self.modules_to_save.items():
            if name in ck:
                try:
                    module.load_state_dict(ck[name], strict=False)
                except Exception:
                    module.load_state_dict(ck[name])
        if load_optimizers:
            if "generator_optimizer" in ck:
                self.generator_optimizer.load_state_dict(
                    ck["generator_optimizer"])
            if "disc_optimizer" in ck:
                self.disc_optimizer.load_state_dict(ck["disc_optimizer"])
        return ck
