from typing import Tuple
import torch
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
    if isinstance(images, Image.Image):
        images = prepare_image(images, input_size).unsqueeze(0)

    elif isinstance(images, torch.Tensor):
        if images.ndim == 3:      # C,H,W
            images = images.unsqueeze(0)
        elif images.ndim != 4:
            raise ValueError("Tensor must be CHW or BCHW")

    else:
        raise TypeError("Input must be PIL.Image or Tensor")

    images = images.to(device)

    outputs = model(images)[0]            # B x num_classes x H x W
    predicted = outputs.argmax(dim=1)     # B x H x W

    mask = (predicted == class_idx)
    mask = mask.unsqueeze(1).to(torch.uint8)  # B x 1 x H x W

    return mask


if __name__ == "__main__":
    from loaders.loader import load_models
    M_C = load_models("M_C", pretrained=True)
