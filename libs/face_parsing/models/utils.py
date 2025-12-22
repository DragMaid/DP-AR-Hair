import torch
import torchvision.transforms as T

normalize_image = T.Compose([
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# TODO: add test this for this heplper function
@torch.no_grad()
def get_mask_by_idx(
    images: torch.Tensor,      # B x 3 x H x W
    model,
    device="cpu",
    class_idx: int = 17,
):
    images = images.to(device)
    images = normalize_image(images)

    masks = []

    for img in images:              # img: 3 x H x W
        img = img.unsqueeze(0)      # 1 x 3 x H x W

        logits = model(img)[0]      # 1 x C x H x W
        pred = logits.argmax(dim=1)  # 1 x H x W

        mask = (pred == class_idx).to(torch.uint8)  # 1 x H x W
        masks.append(mask)

    return torch.stack(masks, dim=0)  # B x 1 x H x W
