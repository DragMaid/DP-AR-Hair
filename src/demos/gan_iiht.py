from loaders.downloader import download_weights
import os
import sys
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from loaders.loader import load_models, ModelRegistry

input_path = Path(__file__).resolve().parents[1] / "assets/test_images/"

face = "phuc.jpeg"
shape = "ken.png"
color = "ken.png"

# The model load weights by itself, ensure pretrained is always set to False
name = "IIHT1"

record = ModelRegistry.get_registry(name)
w_options = record["weight"]["options"]
dest = w_options["local_dir"] / w_options["allow_patterns"][0].split("/")[0]
if not dest.exists():
    download_weights(record["weight"]["type"], w_options)

model = load_models(name, pretrained=False)


def convert_input(inp: str):
    """Load an image from assets/test_images/ as a PIL Image."""
    path = os.path.join(input_path, inp)
    try:
        if os.path.isfile(path):
            return Image.open(path)
        else:
            print(f"File not found: {path}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Can't open the image {inp}: {e}", file=sys.stderr)
        return None


def ensure_pil(img):
    """Ensure the output is a PIL Image (tensor → PIL)."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        return T.functional.to_pil_image(img)
    raise TypeError(f"Unsupported output image type: {type(img)}")


def save_image(img, filename="result.png"):
    """Save an image to ./results/ folder."""
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / filename
    img.save(out_path)
    print(f"Saved result to {out_path}", file=sys.stderr)


converted_inputs = list(map(convert_input, (face, shape, color)))

# If any image failed to load, stop early
if any(img is None for img in converted_inputs):
    print("One or more input images failed to load.", file=sys.stderr)
    sys.exit(1)

face_img, shape_img, color_img = converted_inputs

result_image = model(face_img, shape_img, color_img)

result_image = ensure_pil(result_image)
save_image(result_image, "output.png")
