import os
import torch
import datetime
from pathlib import Path
from math import ceil
from collections import defaultdict
from torchvision.utils import save_image
from losses.adversarial_loss import PatchGANDiscriminator, weights_init
from face_parsing.models.utils import get_mask_by_idx
from configs.pipeline_config import pipeline_config as pco
from loaders.loader import load_models
from models.msg_spade_decoder import MSGSpadeDecoder
from losses.loss_handler import LossHandler
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainingPipeline:
    """
    Extended TrainingPipeline:
    - provides train_step(batch, device, scaler=None, reference_mode='self')
    - provides save_checkpoint / load_checkpoint
    - collects a minimal set of modules to save
    """

    def __init__(self, device, local_rank=None, loaded=True):
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
        # D_S will load and freeze all params except for GF_SPADEs
        self.D_S = load_models("D_S", pretrained=loaded,
                               strict=False, freeze=True).to(self.device)
        self.D_C = load_models("D_C", pretrained=loaded,
                               freeze=True).to(self.device)

        # Wrapped in DDP for distributed parallel training
        self.E_C = DDP(self.E_C, device_ids=[local_rank], output_device=local_rank) if (
            device.type == "cuda") else DDP(self.E_C)
        self.D_S = DDP(self.D_S, device_ids=[local_rank], output_device=local_rank) if (
            device.type == "cuda") else DDP(self.D_S)

        self.D = MSGSpadeDecoder(self.D_C, self.D_S)

        self.generator_trainable_params = []
        # include any parameters that require grad from D_S and E_C
        self.generator_trainable_params += [
            p for p in self.D_S.parameters() if p.requires_grad]
        self.generator_trainable_params += [
            p for p in self.E_C.parameters() if p.requires_grad]

        # Refering to entire pipeline beside IIHT
        self.generator_optimizer = torch.optim.Adam(
            self.generator_trainable_params,
            lr=pco.training.generator.learn_rate,
            betas=pco.training.generator.betas)

        # --- Adversarial discriminator ---
        self.L_adv = PatchGANDiscriminator(n_in_channels=3).to(self.device)
        self.L_adv.apply(weights_init)
        self.L_adv = DDP(self.L_adv, device_ids=[local_rank], output_device=local_rank) if (
            device.type == "cuda") else DDP(self.L_adv)

        # Discrimination optmizer
        self.disc_optimizer = torch.optim.Adam(
            self.L_adv.parameters(),
            lr=pco.training.discriminator.learn_rate,
            betas=pco.training.discriminator.betas)

        # --- Adversarial discriminator ---
        self.losses = LossHandler(self.device)

        # --- Modules to save ---
        self.modules_to_save = {
            "E_C": self.E_C,
            "D_S": self.D_S,
            "L_adv": self.L_adv
        }

    def train_step(self, I_s, I_d, I_d_dilde, scaler,
                   mini_batch_size,
                   save_debug=False,
                   save_path=Path(".")):
        """
        Perform a single training step (one batch) with gradient accumulation.
        - I_s, I_d, I_d_dilde: tensors (BCHW) on CPU or device
        - scaler: optional torch.cuda.amp.GradScaler for mixed precision
        Returns dict of scalars (floats) for logging.
        """

        I_s_o = I_s.to(self.device)
        I_d_o = I_d.to(self.device)
        I_d_dilde_o = I_d_dilde.to(self.device)

        batch_size = I_s.shape[0]
        assert mini_batch_size <= batch_size

        logs = defaultdict(float)
        steps = ceil(batch_size / mini_batch_size)
        for i in range(steps):
            # This is for gradient accumulation
            start = mini_batch_size * i
            end = min(start + mini_batch_size, batch_size)

            I_s = I_s_o[start:end]
            I_d = I_d_o[start:end]
            I_d_dilde = I_d_dilde_o[start:end]

            # Get mask for hair segment
            with torch.no_grad():
                # class_idx = 17 is used to get hair mask
                m_c = get_mask_by_idx(I_d_dilde, self.M_C,
                                      device=self.device, class_idx=17)
                m_f = 1 - m_c  # Inverted m_c or non-hair binary mask

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                f_h = self.E_H(I_s)
                f_m = self.E_M(I_s)["kp"].view(I_s.size(0), -1, 3)
                f_m_d = self.E_M(I_d)["kp"].view(I_s.size(0), -1, 3)
                f_w = self.W(feature_3d=f_h, kp_source=f_m,
                             kp_driving=f_m_d)['out']
                f_c = self.E_C(I_d_dilde)
                # Final predicted image tensor
                I_p = self.D(f_c, f_w, m_c)

            # WARN: wouldn't this decrease the accuracy
            I_p_detached = I_p.detach().float()
            del I_p, f_c

            # --- Discriminator update ---
            disc_loss = self.losses.compute_discriminator_loss(
                I_d, I_p_detached, self.L_adv)

            # backprop discriminator
            disc_loss = disc_loss / steps
            scaler.scale(disc_loss).backward()

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                f_c = self.E_C(I_d_dilde)
                del I_d_dilde
                # Final predicted image tensor
                I_p = self.D(f_c, f_w, m_c)

            del f_c, f_h, f_m, f_m_d, f_w

            # --- Generator update ---
            for p in self.L_adv.parameters():
                p.requires_grad = False

            losses = self.losses.compute_generator_losses(
                I_d, I_p, m_c, m_f, self.L_adv)
            gen_loss = losses["total_loss"]

            for p in self.L_adv.parameters():
                p.requires_grad = True

            gen_loss = gen_loss / steps

            for k, v in losses.items():
                logs[k] += float(v.detach().cpu()) / steps
            del I_d, I_p

            # Backward the generator
            scaler.scale(gen_loss).backward()

            logs["disc_loss"] += float(disc_loss.detach().cpu())
            del losses, disc_loss

        # Update the weights and reset optimizer
        scaler.step(self.disc_optimizer)
        scaler.step(self.generator_optimizer)
        self.disc_optimizer.zero_grad(set_to_none=True)
        self.generator_optimizer.zero_grad(set_to_none=True)
        scaler.update()

        # Save image for debug purposes
        if save_debug:
            img = I_p_detached[0]
            img = (img + 1) / 2
            os.makedirs(save_path, exist_ok=True)
            path = Path.joinpath(
                save_path, f"{datetime.datetime.now()}.png")
            save_image(img, path)
            print(f"Debug image saved to {path}")

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
