import sys
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from loaders.utils import load_hfg_generator


input_path = Path(__file__).resolve().parents[2] / "assets/test_images/"

face = "phuc.jpeg"
shape = "ken.png"
color = "ken.png"
side = "cropped.png"

ALIGNMENT_MODE = "Auto"  # Auto, On, Off
SAVE_PATH = "results/output.png"
SAVE_DIR = "assets/results/"

model = load_hfg_generator()
path_to_imgs = {}


def convert_input(inp):
    """Load local image from INPUT_DIR. Cache images."""
    path = input_path / inp
    if not path.is_file():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return None
    if path in path_to_imgs:
        return path_to_imgs[path]
    try:
        img = Image.open(path).convert("RGB")
        path_to_imgs[path] = img
        return img
    except Exception as e:
        print(f"[ERROR] Can't open image {inp}: {e}", file=sys.stderr)
        return None


def save_output(img, path):
    """Save PIL.Image or torch.Tensor to file."""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(img, torch.Tensor):
        img = TF.to_pil_image(img.clamp(0, 1))
    img.save(path)
    print(f"[INFO] Saved output to {path}", file=sys.stderr)


converted_inputs = list(map(convert_input, (face, shape, color, side)))
if not all(converted_inputs):
    print("[ERROR] Failed to load input images.", file=sys.stderr)
    sys.exit(1)

face_obj, shape_obj, color_obj, face_side_obj = converted_inputs

need_alignment = any(img.size != (1024, 1024) for img in converted_inputs)
perform_align = ALIGNMENT_MODE == "On" or (
    ALIGNMENT_MODE == "Auto" and need_alignment)

if perform_align:
    result = model(
        face_img=face_obj,
        shape_img=shape_obj,
        color_img=color_obj,
        side_face_img=face_side_obj,
        align=True
    )

    generated_save_path = Path(SAVE_DIR, "generated.jpg")
    driving_save_path = Path(SAVE_DIR, "driving.jpg")
    reference_save_path = Path(SAVE_DIR, "reference.jpg")

    save_output(result["final_image"], generated_save_path)
    save_output(result["aligned_face"], driving_save_path)
    save_output(result["aligned_face_side"], reference_save_path)
else:
    result_image = model(face_obj, shape_obj, color_obj)
    save_output(result_image, SAVE_PATH)
