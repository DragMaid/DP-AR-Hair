import random
import torch
import cv2
import dlib
import numpy as np
import argparse
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from torchvision import transforms as T
from loaders.loader import load_models
from face_parsing.models.utils import normalize_image
from data.celebvhq_reference import CelebVHQReferenceDataset
from configs.pipeline_config import pipeline_config
from hair_gan.utils.shape_predictor import get_landmark_detector

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


class Validator:
    def __init__(self, cache_path: str, driving_dir: str, reference_dir: str):
        self.cache_path = cache_path
        self.driving_dir = driving_dir
        self.reference_dir = reference_dir

        self.masker = load_models("M_C", pretrained=True, freeze=True)
        self.landmarker = get_landmark_detector()
        self.detector = dlib.get_frontal_face_detector()

    def validate_image(self, path: str, view=False, landmark=True) -> bool:
        image_ori = cv2.imread(path)
        image = transform(image_ori)
        gray = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)

        image = normalize_image(image)
        image = image.unsqueeze(0)

        logits = self.masker(image)[0]
        prediction = logits.argmax(dim=1).float()

        prediction = prediction.unsqueeze(0)
        prediction = F.interpolate(
            prediction, size=(DOWNSCALE_SIZE, DOWNSCALE_SIZE), mode="nearest")

        prediction = torch.squeeze(prediction)
        prediction_flat = torch.flatten(prediction).tolist()

        counter = Counter(prediction_flat)
        if view:
            self.show_mask_overlay(image_ori, prediction)

        if (counter[18] / (DOWNSCALE_SIZE ** 2) * 100) > INVALID_THRESHOLD:
            return False

        if (counter[17] / (DOWNSCALE_SIZE ** 2) * 100) < INVALID_THRESHOLD:
            return False

        if landmark:
            detections = self.detector(gray, 1)
            if len(detections) == 0:
                return False

            detections = sorted(detections, key=lambda detections: detections.width()
                                * detections.height(), reverse=True)
            try:
                self.landmarker(gray, detections[0])
            except RuntimeError as e:
                print(f"An error occurred: {e}")
                return False

        return True

    def validate_dataset(self):
        dataset = CelebVHQReferenceDataset(
            driving_dir=self.driving_dir,
            reference_dir=self.reference_dir,
            transform=transform
        )

        with open(self.cache_path, "a") as f:
            count = 0
            total = len(dataset)

            reference_paths = list(
                Path(self.reference_dir).glob("*.[jp][pn]g"))

            for combinations in dataset:
                print(f"Processing {count} / {total} items ...")
                count += 1

                if not self.validate_image(combinations["front"]["path"]):
                    continue

                # Side images do not need landmark check
                if not self.validate_image(combinations["side"]["path"], landmark=False):
                    continue

                ref_path = random.choice(reference_paths)

                while not self.validate_image(str(ref_path)):
                    ref_path = random.choice(reference_paths)

                f.write(
                    f"{combinations['front']['path']},{combinations['side']['path']},{ref_path}\n")

    def permute_till(self, goal: int, append_patch_mode: bool = False):
        with open(self.cache_path, "r") as f:
            lines = f.readlines()
            cache_combs = set()
            for line in lines:
                record = line.strip().split(',')
                cache_combs.add((record[0], record[-1]))
            original_len = len(lines)

        with open(self.cache_path, "a") as f:
            # Patch mode also added for quick db insertion
            patch = open("patch.txt", "a") if append_patch_mode else None

            added_count = 0

            reference_paths = list(
                Path(self.reference_dir).glob("*.[jp][pn]g"))

            while original_len + added_count < goal:
                print(f"Processing {original_len + added_count} / {goal} ...")
                line = random.choice(lines)
                drive_front_path, drive_side_path, _ = line.strip().split(',')

                ref_path = random.choice(reference_paths)
                current_comb = (drive_front_path, ref_path)

                while current_comb not in cache_combs and \
                        not self.validate_image(str(ref_path)):
                    ref_path = random.choice(reference_paths)

                cache_combs.add(current_comb)
                f.write(
                    f"{drive_front_path},{drive_side_path},{ref_path}\n")
                patch.write(
                    f"{drive_front_path},{drive_side_path},{ref_path}\n")

                added_count += 1

            if patch:
                patch.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, help="Path to cache file",
                        default=pipeline_config.generation.cache_path)
    parser.add_argument("--reference", type=str, help="Path to reference images folder",
                        default=pipeline_config.generation.reference_dir)
    parser.add_argument("--driving", type=str, help="Path to driving images folder",
                        default=pipeline_config.generation.driving_dir)
    parser.add_argument("--size", type=int, help="Dataset size to generate",
                        default=20_000)
    parser.add_argument("--patch", type=bool, help="Patch mode to output a patch file",
                        default=False)

    args = parser.parse_args()
    validator = Validator(
        cache_path=args.cache,
        reference_dir=args.reference,
        driving_dir=args.driving
    )

    # If cache file is not created yet then re-run whole validation
    if not Path(args.cache).exists():
        validator.validate_dataset()
    validator.permute_till(goal=args.size, append_patch_mode=args.patch)
