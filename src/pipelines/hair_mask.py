import os
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from models.bisenet import BiSeNet


def prepare_image(image: Image.Image, input_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
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


def load_model(model_name: str, num_classes: int, weight_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load BiSeNet model with pretrained weights.
    """
    model = BiSeNet(num_classes, backbone_name=model_name)
    model.to(device)

    if not os.path.exists(weight_path):
        raise ValueError(f"Weights not found at {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def get_hair_mask(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    input_size: Tuple[int, int] = (512, 512),
    hair_class_idx: int = 17  # BiSeNet class index for hair
) -> np.ndarray:
    """
    Run inference on a single image and return the binary hair mask.

    Returns:
        np.ndarray: Binary mask of shape (H, W), 1 for hair, 0 for non-hair.
    """
    original_size = image.size  # (width, height)
    image_batch = prepare_image(image, input_size).to(device)

    output = model(image_batch)[0]  # use main output only
    predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

    # Hair mask only
    hair_mask = (predicted_mask == hair_class_idx).astype(np.uint8)

    # Resize back to original image resolution
    hair_mask_pil = Image.fromarray(hair_mask * 255)  # multiply by 255 for PIL
    hair_mask_resized = hair_mask_pil.resize(
        original_size, resample=Image.NEAREST)

    return np.array(hair_mask_resized) // 255  # return 0/1 mask
