import os
import requests
import sys
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms.functional as TF

from loaders.downloader import download_weights
from loaders.loader import load_models, ModelRegistry

from httpx import AsyncClient
from manager.client.core.session import session
from manager.client.api.fetcher import APIFetcher
from manager.client.core.config import settings
from manger.client.api.tasks import claim_task
from manager.client.api.auth import authorize
from manager.schemas.user import UserRoles
from manager.schemas.image import ImageCategories
from manager.schemas.assignment import AssignmentStatus
from manager.client.api.image import upload
from manager.client.api.assignment import report_assignment

SAVE_DIR = "./results/"


class Worker:

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

        # self.model = self.init_generator()

        self.client = AsyncClient()
        self.fetcher = APIFetcher(
            base_url=settings.BASE_URL,
            client=self.client,
            session=session,
            strict=False
        )

    def init_generator(self):
        """Initialize the weights and return the generator instance."""

        name = "IIHT1"
        record = ModelRegistry.get_registry(name)
        w_options = record["weight"]["options"]
        dest = w_options["local_dir"] / \
            w_options["allow_patterns"][0].split("/")[0]

        if not dest.exists():
            download_weights(record["weight"]["type"], w_options)

        # The model load weights by itself so pretrained is False
        return load_models(name, pretrained=False)

    def convert_input(self, file_path: str):
        """Load local image from INPUT_DIR. Cache images."""

        file_path = Path(file_path)
        if not file_path.is_file():
            print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
            return None

        try:
            img = Image.open(file_path).convert("RGB")
            return img
        except Exception as e:
            print(
                f"[ERROR] Can't open image {file_path}: {e}", file=sys.stderr)
            return None

    def save_output(self, img, path: Path):
        """Save PIL.Image or torch.Tensor to file."""

        # os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img.clamp(0, 1))
        img.save(path)
        print(f"[INFO] Saved output to {path}", file=sys.stderr)

    async def claim_task(self):
        if not session.is_authenticated():
            authorize(
                fetcher=self.fetcher,
                username=self.username,
                password=self.password,
                role=UserRoles.WORKER
            )

        response = await claim_task(self.fetcher)
        driving_path = response["driving_path"]
        reference_path = response["reference_path"]

        self.assignment_id = response["assignment_id"]
        self.download_image(driving_path, "assets/tmp/driving.jpg")
        self.download_image(reference_path, "assets/tmp/reference.jpg")

    def download_image(self, source_path, output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        image = requests.get(os.path.join(
            settings.BASE_URL, source_path)).content
        with open(output_path, 'wb') as handler:
            handler.write(image)

    def inference(
        self,
        face_path: str,
        shape_path: str,
        color_path: str
    ):
        converted_inputs = list(
            map(
                self.convert_input,
                (face_path, shape_path, color_path)
            )
        )
        if not all(converted_inputs):
            print("[ERROR] Failed to load input images.", file=sys.stderr)
            raise

        face_obj, shape_obj, color_obj = converted_inputs

        # Always perform alignment
        result_image, face_obj, shape_obj, color_obj = self.model(
            face_obj, shape_obj, color_obj, align=True)

        # TODO: re-write so it saves aligned faces also
        self.save_output(result_image, "assets/generated_images/generated.jpg")

    async def report(self):
        driving_id = await upload(
            fetcher=self.fetcher,
            assignment_id=self.assignment_id,
            path=Path(SAVE_DIR, "driving.jpg"),
            category=ImageCategories.DRIVING
        )
        reference_id = await upload(
            fetcher=self.fetcher,
            assignment_id=self.assignment_id,
            path=Path(SAVE_DIR, "reference.jpg"),
            category=ImageCategories.REFERENCE
        )
        generated_id = await upload(
            fetcher=self.fetcher,
            assignment_id=self.assignment_id,
            path=Path(SAVE_DIR, "generated.jpg"),
            category=ImageCategories.GENERATED
        )

        await report_assignment(
            fetcher=self.fetcher,
            assignment_id=self.assignment_id,
            driving_upload_id=driving_id,
            reference_upload_id=reference_id,
            generated_upload_id=generated_id,
            status=AssignmentStatus.SUCCEED
        )

    def run(self):
        pass


if __name__ == "__main__":
    worker = Worker(
        username="bob",
        password="dwS4t6JSMXvYHk0Obv4NgHA7lMhmub-SC"
    )
    worker.run()
