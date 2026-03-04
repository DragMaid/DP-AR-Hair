import os
import torch
from math import ceil
from losses.adversarial_loss import PatchGANDiscriminator
from face_parsing.models.utils import get_mask_by_idx
from loaders.loader import load_models
from models.msg_spade_decoder import MSGSpadeDecoder
from losses.loss_handler import LossHandler
from torch.nn.parallel import DistributedDataParallel as DDP
from hairshifter.utils import discriminator_augment_pair, jitter_binary_mask
from configs.pipeline_config import pipeline_config as pco
from hairshifter.utils import init_weights


def nan_hook(name):
    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            if not torch.isfinite(out).all():
                raise RuntimeError(f"NaN/Inf detected in {name}")
    return hook


class TrainingPipeline:
    """
    Extended TrainingPipeline:
    - provides train_step(batch, device, scaler=None, reference_mode='self')
    - provides save_checkpoint / load_checkpoint
    - collects a minimal set of modules to save
    """

    def __init__(self, device, logger, local_rank=None, loaded=True):
        self.device = device
        self.logger = logger

        # --- Generator (G) ---
        self.E_H = load_models("E_H", pretrained=loaded,
                               freeze=True).to(self.device)
        self.E_M = load_models("E_M", pretrained=loaded,
                               freeze=True).to(self.device)
        self.E_C = load_models("E_C", pretrained=False
                               ).to(self.device)  # trainable
        self.W = load_models("W", pretrained=loaded,
                             freeze=True).to(self.device)
        self.M_C = load_models("M_C", pretrained=True,
                               freeze=True).to(self.device)

        # D_S will load and freeze all params except for GF_SPADEs
        self.D_S = load_models("D_S", pretrained=loaded,
                               freeze=True, strict=False).to(self.device)
        self.D_C = load_models("D_C", pretrained=loaded,
                               freeze=True, strict=False).to(self.device)

        for name, module in self.E_C.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                module.register_forward_hook(nan_hook(name))

        # Wrapped in DDP for distributed parallel training
        init_weights(self.E_C)
        self.E_C = DDP(self.E_C, device_ids=[local_rank], output_device=local_rank) if (
            device.type == "cuda") else DDP(self.E_C)

        self.D_S = DDP(self.D_S, device_ids=[local_rank], output_device=local_rank) if (
            device.type == "cuda") else DDP(self.D_S)

        self.D = MSGSpadeDecoder(self.D_C, self.D_S)

        for name, module in self.D.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                module.register_forward_hook(nan_hook(name))

        # include any parameters that require grad from D_S and E_C
        self.synthesizer_decoder_trainable_dict = {
            k: p for k, p in self.D_S.named_parameters() if p.requires_grad}
        self.context_encoder_trainable_dict = {
            k: p for k, p in self.E_C.named_parameters() if p.requires_grad}
        self.generator_trainable_dict = self.synthesizer_decoder_trainable_dict | \
            self.context_encoder_trainable_dict

        self.synthesizer_decoder_trainable_params = list(
            self.synthesizer_decoder_trainable_dict.values())
        self.context_encoder_trainable_params = list(
            self.context_encoder_trainable_dict.values())
        self.generator_trainable_params = self.synthesizer_decoder_trainable_params + \
            self.context_encoder_trainable_params

        # --- Adversarial discriminator ---
        self.L_adv = PatchGANDiscriminator(n_in_channels=3).to(self.device)
        init_weights(self.L_adv)

        self.L_adv = DDP(self.L_adv, device_ids=[local_rank], output_device=local_rank) if (
            device.type == "cuda") else DDP(self.L_adv)
        self.disc_trainable_params = [
            p for p in self.L_adv.parameters() if p.requires_grad]

        # --- Adversarial discriminator ---
        self.losses = LossHandler(self.device)

        # --- Modules to save ---
        self.modules_to_save = {
            "E_C": self.E_C,
            "D_S": self.D_S,
            "L_adv": self.L_adv
        }

        # --- Modules to log ---
        self.modules_to_log = {
            "generator": self.generator_trainable_params,
            "discriminator": self.disc_trainable_params,
            "context_encoder": self.context_encoder_trainable_params,
            "synthesize_decoder": self.synthesizer_decoder_trainable_params,
        }

    def set_optimizers(self, generator_optimizer, disc_optimizer, ema):
        self.generator_optimizer = generator_optimizer
        self.disc_optimizer = disc_optimizer
        self.logger.set_optimizer(gen_optimizer=generator_optimizer)
        self.ema = ema

    def train_step(
        self,
        I_s, I_d, I_d_dilde,
        scaler,
        mini_batch_size,
        freeze_discriminator=False,
        freeze_generator=False,
        accumulate_grad_contrib=False,
        store_outputs=False,
    ):
        """
        Perform a single training step (one batch) with gradient accumulation.
        - I_s, I_d, I_d_dilde: tensors (BCHW) on CPU or device
        - scaler: optional torch.cuda.amp.GradScaler for mixed precision
        Returns dict of scalars (floats) for logging.
        """

        I_s_o = I_s.to(self.device)
        I_d_o = I_d.to(self.device)
        I_d_dilde_o = I_d_dilde.to(self.device)

        self.logger.reset()

        batch_size = I_s.shape[0]
        assert mini_batch_size <= batch_size

        steps = ceil(batch_size / mini_batch_size)
        for i in range(steps):
            # This is for gradient accumulation
            start = mini_batch_size * i
            end = min(start + mini_batch_size, batch_size)
            last_mini = i == steps - 1

            I_s = I_s_o[start:end]
            I_d = I_d_o[start:end]
            I_d_dilde = I_d_dilde_o[start:end]

            # Get mask for hair segment
            with torch.no_grad():
                # class_idx = 17 is used to get hair mask
                m_c = get_mask_by_idx(I_d_dilde, self.M_C,
                                      device=self.device, class_idx=17)

                # Jitter the mask so disc don't get sensitive with sharp edges
                m_c = jitter_binary_mask(
                    m_c, p=pco.training.stablizers.mask_jitter_prob)
                m_f = 1 - m_c  # Inverted m_c or non-hair binary mask

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                f_h = self.E_H(I_s)
                f_m = self.E_M(I_s)["kp"].view(I_s.size(0), -1, 3)
                f_m_d = self.E_M(I_d)["kp"].view(I_s.size(0), -1, 3)
                f_w = self.W(feature_3d=f_h, kp_source=f_m,
                             kp_driving=f_m_d)['out']

                # Calculated only for discriminator training and not re-used
                with torch.no_grad():
                    f_c = self.E_C(I_d_dilde)
                    I_p = self.D(f_c, f_w, m_c)

            I_p_detached = I_p.detach().float()
            del I_p, f_c

            # If discriminator is frozen then no need to train beforehand
            if not freeze_discriminator:
                for p in self.L_adv.parameters():
                    p.requires_grad = True

                I_d_aug, I_p_aug = discriminator_augment_pair(
                    I_d, I_p_detached, p=pco.training.stablizers.image_aug_prob
                )

                # --- Discriminator update ---
                disc_loss = self.losses.compute_discriminator_loss(
                    I_d_aug, I_p_aug, self.L_adv)
                del I_d_aug, I_p_aug

                # backprop discriminator
                scaler.scale(disc_loss / steps).backward()
                self.logger.accumulate_loss(
                    name="discriminator_loss", value=disc_loss)

                del disc_loss
            else:
                for p in self.L_adv.parameters():
                    p.requires_grad = False

            # --- Generator update ---
            if not freeze_generator:
                for p in self.generator_trainable_params:
                    p.requires_grad = True

                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    f_c = self.E_C(I_d_dilde)
                    I_p = self.D(f_c, f_w, m_c)

                del I_d_dilde, f_c, f_h, f_m, f_m_d, f_w

                for p in self.L_adv.parameters():
                    p.requires_grad = False

                gen_losses = self.losses.compute_generator_losses(
                    I_d, I_p, m_c, m_f, self.L_adv)

                gen_loss = gen_losses["generator_loss"]
                for k, v in gen_losses.items():
                    self.logger.accumulate_loss(name=k, value=v)

                scaler.scale(
                    gen_loss / steps).backward(retain_graph=accumulate_grad_contrib)

                if accumulate_grad_contrib and last_mini:
                    self.logger.calculate_loss_contribution(
                        gen_losses=gen_losses,
                        params=self.generator_trainable_params,
                        scaler=scaler,
                    )

                del I_d, I_p, gen_losses

                for p in self.L_adv.parameters():
                    p.requires_grad = True
            else:
                for p in self.generator_trainable_params:
                    p.requires_grad = False

            if store_outputs:
                self.logger.log_images(I_p_detached)

            self.logger.step_done()

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
        payload["ema"] = self.ema.shadow

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

            if "ema" in ck and self.ema.enabled:
                for name, param in self.generator_trainable_dict.items():
                    if param.requires_grad and name in ck['ema']:
                        self.ema.shadow[name] = ck['ema'][name].detach().cpu().clone()

        return ck
