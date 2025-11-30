import torch
import numpy as np
import torchvision.transforms as transforms
from typing import Tuple
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
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    input_size: Tuple[int, int] = (512, 512),
    class_idx: int = 17  # BiSeNet class index for hair
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
    mask = (predicted_mask == class_idx).astype(np.uint8)

    # Resize back to original image resolution
    mask_pil = Image.fromarray(mask * 255)  # multiply by 255 for PIL
    mask_resized = mask_pil.resize(
        original_size, resample=Image.NEAREST)

    return np.array(mask_resized) // 255  # return 0/1 mask


if __name__ == "__main__":
    from loaders.loader import load_models
    M_C = load_models("M_C", pretrained=True)
