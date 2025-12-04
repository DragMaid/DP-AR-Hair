from typing import Union, List, Tuple
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def prepare_image(image: Image.Image,
                  input_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Resize and normalize the image for the model.
    """
    resized_image = image.resize(input_size, resample=Image.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_tensor = transform(resized_image).unsqueeze(0)
    return image_tensor


@torch.no_grad()
def get_mask_by_idx(
    images,
    model,
    device,
    input_size=(512, 512),
    class_idx=17
):
    """
    Supports:
    - PIL.Image (single)
    - torch.Tensor CxHxW (single)
    - torch.Tensor BxCxHxW (batch)
    Returns binary mask: BxHxW (torch.uint8)
    """
    if isinstance(images, Image.Image):
        images = prepare_image(images, input_size)
    elif isinstance(images, torch.Tensor):
        if images.ndim == 3:  # C,H,W
            images = images.unsqueeze(0)
        elif images.ndim != 4:
            raise ValueError("Tensor must be CHW or BCHW")
    else:
        raise TypeError("Input must be PIL.Image or Tensor")

    images = images.to(device)  # move to GPU
    B, _, _, _ = images.shape

    outputs = model(images)[0]          # outputs: B x num_classes x H x W

    predicted = outputs.argmax(dim=1)   # B x H x W

    mask = (predicted == class_idx).to(torch.uint8)  # B x H x W

    resized_masks = []
    for i in range(B):
        orig_w, orig_h = get_image_size(images[i])
        pil_mask = Image.fromarray(mask[i].cpu().numpy() * 255)
        resized = pil_mask.resize((orig_w, orig_h), Image.NEAREST)
        resized_masks.append(torch.from_numpy(
            np.array(resized) // 255))

    # Stack → B x 1 x H_orig x W_orig
    return torch.stack(resized_masks, dim=0).unsqueeze(1)


def get_image_size(image):
    if isinstance(image, Image.Image):
        return image.size  # (W, H)

    if isinstance(image, torch.Tensor):
        # Tensor layout: C,H,W
        _, h, w = image.shape
        return (w, h)

    raise TypeError("Unsupported image type")


if __name__ == "__main__":
    from loaders.loader import load_models
    M_C = load_models("M_C", pretrained=True)
