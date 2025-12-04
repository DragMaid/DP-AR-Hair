import torch


class HairFastBatchWrapper:
    """
    Wraps HairFast to process a batch of images sequentially.
    """

    def __init__(self, hairfast_model, device=None):
        """
        hairfast_model: instance of HairFast
        device: torch.device or None (defaults to 'cuda:0' if available)
        """
        self.model = hairfast_model
        self.device = device or torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def batch_swap(self, face_batch, shape_batch, color_batch, **kwargs):
        """
        face_batch, shape_batch, color_batch: tensors [B,C,H,W]
        Returns tensor [B,C,H,W] with processed images.
        """
        results = []

        for i in range(face_batch.size(0)):
            face_img = face_batch[i].to(self.device)
            shape_img = shape_batch[i].to(self.device)
            color_img = color_batch[i].to(self.device)

            need_alignment = any(img.size != (1024, 1024)
                                 for img in (face_img, shape_img, color_img))

            if need_alignment:
                out, _, _, _ = self.model.swap(
                    face_img, shape_img, color_img, align=True, **kwargs)
            else:
                out = self.model.swap(face_img, shape_img, color_img, **kwargs)

            # handle the rare 4D single-image output
            if out.ndim == 4 and out.size(0) == 1:
                out = out.squeeze(0)

            results.append(out.cpu())

        return torch.stack(results, dim=0)  # [B,C,H,W]
