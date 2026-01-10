import random
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms as T
from loaders.loader import load_models
from face_parsing.models.utils import normalize_image
from collections import Counter
from PIL import Image
from data.celebvhq_reference import CelebVHQReferenceDataset
from configs.pipeline_config import pipeline_config
from pathlib import Path

# 5% will be the threshold for both hat and bald
UPSCALE_SIZE = 512
DOWNSCALE_SIZE = 256
INVALID_THRESHOLD = 5
SHOW_OVERLAY = True

transform = T.Compose([
    T.ToPILImage(),
    T.Resize([UPSCALE_SIZE, UPSCALE_SIZE]),
    T.ToTensor()
])

masker = load_models("M_C", pretrained=True, freeze=True)


def validate_image(path: str, view=False) -> bool:
    image_ori = cv2.imread(path)
    image = transform(image_ori)

    image = normalize_image(image)
    image = image.unsqueeze(0)

    logits = masker(image)[0]
    prediction = logits.argmax(dim=1).float()

    prediction = prediction.unsqueeze(0)
    prediction = F.interpolate(
        prediction, size=(DOWNSCALE_SIZE, DOWNSCALE_SIZE), mode="nearest")

    prediction = torch.squeeze(prediction)
    prediction_flat = torch.flatten(prediction).tolist()

    counter = Counter(prediction_flat)
    if view:
        show_mask_overlay(image_ori, prediction)

    if (counter[18] / (DOWNSCALE_SIZE ** 2) * 100) > INVALID_THRESHOLD:
        return False

    if (counter[17] / (DOWNSCALE_SIZE ** 2) * 100) < INVALID_THRESHOLD:
        return False

    return True


def show_mask_overlay(image_ori: Image, prediction: torch.Tensor):
    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    image_ori = cv2.resize(image_ori, (DOWNSCALE_SIZE, DOWNSCALE_SIZE))

    num_categories = 21
    alpha = 0.4  # For transparency
    colors = {}
    for index in range(num_categories):
        colors[index] = [random.randint(0, 255) for _ in range(3)]

    color_mask = np.zeros(
        (prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    for cat_id, color in colors.items():
        color_mask[prediction == cat_id] = color

    overlay = image_ori.copy()
    overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)

    plt.imshow(overlay)
    plt.axis("off")
    plt.show()


def validate_dataset():
    dataset = CelebVHQReferenceDataset(
        driving_dir=pipeline_config.dataset.driving_dir,
        reference_dir=pipeline_config.dataset.reference_dir,
        transform=transform
    )

    with open("cache.txt", "a") as f:
        count = 0
        total = len(dataset)
        for combinations in dataset:
            print(f"Processing {count} / {total} items ...")
            count += 1

            if not validate_image(combinations["front"]["path"]):
                continue

            if not validate_image(combinations["side"]["path"]):
                continue

            reference_paths = list(
                Path(pipeline_config.dataset.reference_dir).glob("*.[jp][pn]g"))
            ref_path = random.choice(reference_paths)

            while not validate_image(str(ref_path)):
                ref_path = random.choice(reference_paths)

            f.write(
                f"{combinations['front']['path']},{combinations['side']['path']},{ref_path}\n")


def test_dangerous_images():
    DANGEROUS_IMAGES = [
        "0bR6pUOhZo4_2_frontal",
        "-0fMjAGBbuE_17_0_frontal",
        "-2Xf6uifdt8_19_0_frontal",
        "-4SwiOfkvuA_0_0_frontal",
        "-4SwiOfkvuA_0_0_side",
        "0blgdAE1cbk_2_0_frontal",
        "0Fxrs1a7fD0_1_1_frontal",
        "1Acvwko6Wd0_0_frontal",
        "1XHdPvd9HPo_15_frontal",
        "3mgh_1-1sTU_72_0_frontal",
        "6p_LlhzOrBk_7_frontal",
        "1akcYVlAvjE_10_1_frontal",
        "1diHvu1Q6Ec_1_frontal",
        "2kfb-E2OXAk_11_0_side",
    ]
    for name in DANGEROUS_IMAGES:
        validate_image(f"assets/driving_images/{name}.jpg", view=True)


if __name__ == "__main__":
    validate_dataset()
