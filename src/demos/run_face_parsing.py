from pathlib import Path
from torchvision import transforms as T
from loaders.loader import load_models
from face_parsing.models.utils import get_mask_by_idx
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 17 is for face
# 18 is for hat

transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])

masker = load_models("M_C", pretrained=True, freeze=True)

path = Path("./DP-AR-Hair_Output/")

for p in path.iterdir():
    # ---- load & preprocess image (your original flow) ----
    image_bgr = cv2.imread(str(p))
    image = transform(image_bgr)
    image = image.unsqueeze(0)  # Add batching to fit util func

    # ---- get binary mask ----
    mask = get_mask_by_idx(image, masker, class_idx=17)
    mask = mask.squeeze(0).squeeze(0)       # (256,256), {0,1}
    mask_u8 = (mask * 255).byte()           # uint8 {0,255}

    # ---- prepare original image for visualization ----
    orig = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (512, 512))   # (256,256,3), uint8

    # ---- overlay mask with transparency ----
    alpha = 0.4
    color = np.array([255, 0, 0], dtype=np.uint8)  # red overlay

    mask_np = mask_u8.cpu().numpy()
    mask_bool = mask_np > 0

    overlay = orig.copy()
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + color * alpha
    ).astype(np.uint8)

    # ---- show result ----
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()
