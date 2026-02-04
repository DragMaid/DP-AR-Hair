import torch
import torchvision.transforms as T

# NOTE: The model was trained on 512x512 so better keep that to make sure the predictions are accurate

normalize_image = T.Compose([
    T.Resize((512, 512)),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


@torch.no_grad()
def get_mask_by_idx(
    images: torch.Tensor,      # B x 3 x H x W
    model,
    device="cpu",
    class_idx: int = 17,
):
    images = images.to(device)
    orig_size = images[0].shape[1:]
    images = normalize_image(images)

    masks = []

    for img in images:              # img: 3 x H x W
        img = img.unsqueeze(0)      # 1 x 3 x H x W

        logits = model(img)[0]      # 1 x C x H x W
        pred = logits.argmax(dim=1)  # 1 x H x W

        mask = (pred == class_idx).float()  # 1 x H x W
        masks.append(T.functional.resize(mask, orig_size))

    return torch.stack(masks, dim=0)  # B x 1 x H x W
