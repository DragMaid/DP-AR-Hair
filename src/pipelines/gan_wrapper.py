import torch
from itertools import cycle


class HairFastBatchWrapper:
    """
    Wraps HairFast to process a batch of images on multiple GPUs in parallel.
    """

    def __init__(self, hairfast_model, device_ids=None):
        """
        hairfast_model: instance of HairFast
        device_ids: list of GPU ids to use, e.g. [0,1,2]
        """
        self.model = hairfast_model
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids

    @torch.no_grad()
    def batch_swap(self, face_batch, shape_batch, color_batch, **kwargs):
        """
        face_batch, shape_batch, color_batch: tensors of shape [B,C,H,W]
        Returns tensor [B,C,H,W] with processed images
        """
        B = face_batch.size(0)
        results = []

        # cycle through GPUs
        gpu_cycle = cycle(self.device_ids)

        for i in range(B):
            device = torch.device(f"cuda:{next(gpu_cycle)}")

            # HairFast expects 3D tensors
            face_img = face_batch[i].to(device)
            shape_img = shape_batch[i].to(device)
            color_img = color_batch[i].to(device)

            out = self.model.swap(face_img, shape_img, color_img, **kwargs)
            # ensure returned tensor is on CPU
            results.append(out.cpu())

        return torch.stack(results, dim=0)  # [B,C,H,W]
