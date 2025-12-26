import os
import torch
import datetime
from pathlib import Path
from torchvision.utils import save_image
from losses.adversarial_loss import PatchGANDiscriminator, weights_init
from face_parsing.models.utils import get_mask_by_idx
from configs.pipeline_config import pipeline_config as pco
from loaders.loader import load_models, ModelRegistry
from loaders.downloader import download_weights
from models.msg_spade_decoder import MSGSpadeDecoder
from pipelines.gan_wrapper import HairFastBatchWrapper
from losses.loss_handler import LossHandler
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainingPipeline:
    """
    Extended TrainingPipeline:
    - provides train_step(batch, device, scaler=None, reference_mode='self')
    - provides save_checkpoint / load_checkpoint
    - collects a minimal set of modules to save
    """

    def __init__(self, device, local_rank=None, loaded=True, generate_on_go=False):
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
        self.D = MSGSpadeDecoder(self.D_C, self.D_S)

        # Wrapped in DDP for distributed parallel training
        self.E_C = DDP(self.E_C, device_ids=[local_rank], output_device=local_rank) if (
            device.type == "cuda") else DDP(self.E_C)
        self.D = DDP(self.D, device_ids=[local_rank], output_device=local_rank) if (
            device.type == "cuda") else DDP(self.D)

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

        # --- Iterative Implicit Hair Transfer (IIHT) ---
        self.IIHT = None
        if generate_on_go:
            IIHT_NAME = "IIHT1"
            record = ModelRegistry.get_registry(IIHT_NAME)
            w_options = record["weight"]["options"]
            dest = w_options["local_dir"] / \
                w_options["allow_patterns"][0].split("/")[0]
            if not dest.exists():
                download_weights(record["weight"]["type"], w_options)
            self.IIHT = load_models(IIHT_NAME, pretrained=False)
            # TODO: add a more dynamic way to set the device ids
            self.IIHT = HairFastBatchWrapper(self.IIHT)

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

    def train_step(self, I_s, I_d, I_r, scaler,
                   save_debug=False,
                   save_path=Path(".")):
        """
        Perform a single training step (one batch).
        - I_s, I_d, I_r: tensors (BCHW) on CPU or device
        - scaler: optional torch.cuda.amp.GradScaler for mixed precision
        Returns dict of scalars (floats) for logging.
        """

        I_s = I_s.to(self.device)
        I_d = I_d.to(self.device)
        I_r = I_r.to(self.device)

        # TODO: add resizing to 1024x1024 for HairFastGan
        # I_d, I_r, I_s: 4D tensors (B, C, H, W)
        if self.IIHT:
            with torch.no_grad():
                I_d_dilde = self.IIHT.batch_swap(I_d, I_r, I_d)
                I_d_dilde = I_d_dilde.to(self.device)
        else:
            # If the generate_on_go mode is not set then
            # passed in value for I_r would be generated I_d_dilde
            I_d_dilde = I_r

        f_c = self.E_C(I_d_dilde)
        # Other components are just for inference
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                f_h = self.E_H(I_s)
                f_m = self.E_M(I_s)["kp"].view(I_s.size(0), -1, 3)
                f_m_d = self.E_M(I_d)["kp"].view(I_s.size(0), -1, 3)
                f_w = self.W(feature_3d=f_h, kp_source=f_m,
                             kp_driving=f_m_d)['out']
                # Get mask for hair segment
                m_c = get_mask_by_idx(I_d_dilde, self.M_C,
                                      device=self.device, class_idx=17)
                m_f = 1 - m_c  # Inverted m_c or non-hair binary mask
        del I_r, I_d_dilde

        # Final predicted image tensor
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            I_p = self.D(f_c, f_w, m_c)
        I_p_detached = I_p.detach()

        # Save image for debug purposes
        if save_debug:
            img = I_p_detached[0]
            img = (img + 1) / 2
            os.makedirs(save_path, exist_ok=True)
            path = Path.joinpath(save_path, f"{datetime.datetime.now()}.png")
            save_image(img, path)
            print(f"Debug image saved to {path}")

        # --- Discriminator update ---
        self.disc_optimizer.zero_grad(set_to_none=True)
        disc_loss = self.losses.compute_discriminator_loss(
            I_d, I_p_detached, self.L_adv)

        # backprop discriminator
        scaler.scale(disc_loss).backward()
        scaler.step(self.disc_optimizer)

        # --- Generator update ---
        self.generator_optimizer.zero_grad(set_to_none=True)
        losses = self.losses.compute_generator_losses(
            I_d, I_p, m_c, m_f, self.L_adv)
        gen_loss = losses["total_loss"]

        logs = {k: float(v.detach().cpu()) for k, v in losses.items()}
        del I_d, I_p

        # Backward the generator
        scaler.scale(gen_loss).backward()
        scaler.step(self.generator_optimizer)

        # Update the scaler
        scaler.update()

        logs["disc_loss"] = float(disc_loss.detach().cpu())
        del losses, disc_loss

        # return python scalars for logging
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
