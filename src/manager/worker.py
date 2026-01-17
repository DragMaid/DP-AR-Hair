import requests
import sys
from pathlib import Path
from PIL import Image
from urllib.parse import urljoin
from typing import Optional

import torch
import torchvision.transforms.functional as TF

from loaders.downloader import download_weights
from loaders.loader import load_models, ModelRegistry

from httpx import AsyncClient
from manager.client.core.session import session
from manager.client.api.fetcher import APIFetcher
from manager.client.core.config import settings
from manager.client.api.tasks import claim_task
from manager.client.api.auth import authorize
from manager.schemas.user import UserRoles
from manager.schemas.image import ImageCategories
from manager.schemas.assignment import AssignmentStatus
from manager.client.api.image import upload
from manager.client.api.assignment import report_assignment
from manager.client.core.errors import FrontError, ErrorCategories

SAVE_DIR = "./assets/results/"
TMP_DIR = "./assets/tmp/"


class Worker:

    def __init__(
        self,
        username: str,
        password: str,
        base_url: Optional[str] = None
    ):
        self.username = username
        self.password = password

        self.model = self.init_generator()
        self.base_url = base_url if base_url else settings.BASE_URL

        self.client = AsyncClient()
        self.fetcher = APIFetcher(
            base_url=self.base_url,
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

    async def authorize(self):
        await authorize(
            fetcher=self.fetcher,
            username=self.username,
            password=self.password,
            role=UserRoles.WORKER
        )

    # TODO: this is just absurd
    def get_driving_side_path(self, drive_path):
        drive_name = drive_path.split('/')[-1]
        drive_dir = '/'.join(drive_path.split('/')[:-1])
        drive_id = '_'.join(drive_name.split('_')[:-1])
        drive_ext = drive_name.split('.')[-1]
        return f"{drive_dir}/{drive_id}_side.{drive_ext}"

    async def claim_task(self):
        response = await claim_task(self.fetcher)

        driving_path = response["driving_path"]
        reference_path = response["reference_path"]
        driving_side_path = self.get_driving_side_path(driving_path)
        assignment_id = response["assignment_id"]

        driving_save_path = Path(TMP_DIR, "driving.jpg")
        reference_save_path = Path(TMP_DIR, "reference.jpg")
        driving_side_save_path = Path(TMP_DIR, "driving_side.jpg")

        self.download_image(driving_path, driving_save_path)
        self.download_image(reference_path, reference_save_path)
        self.download_image(driving_side_path, driving_side_save_path)

        return {
            "assignment_id": assignment_id,
            "driving_save_path": driving_save_path,
            "driving_side_save_path": driving_side_save_path,
            "reference_save_path": reference_save_path
        }

    def download_image(self, source_path: str, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = urljoin(settings.BASE_URL, source_path)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        output_path.write_bytes(response.content)

    def inference(
        self,
        face_path: str,
        shape_path: str,
        color_path: str,
        face_side_path: str
    ):
        converted_inputs = list(
            map(self.convert_input, (face_path, shape_path, color_path, face_side_path)))

        if not all(converted_inputs):
            print("[ERROR] Failed to load input images.", file=sys.stderr)
            raise

        face_obj, shape_obj, color_obj, face_side_obj = converted_inputs
        # Always perform alignment
        result = self.model(
            face_img=face_obj,
            shape_img=shape_obj,
            color_img=color_obj,
            side_face_img=face_side_obj,
            align=True
        )

        generated_save_path = Path(SAVE_DIR, "generated.jpg")
        driving_save_path = Path(SAVE_DIR, "driving.jpg")
        reference_save_path = Path(SAVE_DIR, "reference.jpg")

        self.save_output(result["final_image"], generated_save_path)
        self.save_output(result["aligned_face"], driving_save_path)
        self.save_output(result["aligned_face_side"], reference_save_path)

        return {
            "generated_path": generated_save_path,
            "driving_path": driving_save_path,
            "reference_path": reference_save_path
        }

    async def report(self, path_map, assignment_id, log, status):
        if status == AssignmentStatus.SUCCEED:
            driving_id = await upload(
                fetcher=self.fetcher,
                assignment_id=assignment_id,
                path=path_map["driving_path"],
                category=ImageCategories.DRIVING
            )
            reference_id = await upload(
                fetcher=self.fetcher,
                assignment_id=assignment_id,
                path=path_map["reference_path"],
                category=ImageCategories.REFERENCE
            )
            generated_id = await upload(
                fetcher=self.fetcher,
                assignment_id=assignment_id,
                path=path_map["generated_path"],
                category=ImageCategories.GENERATED
            )

            await report_assignment(
                fetcher=self.fetcher,
                assignment_id=assignment_id,
                driving_upload_id=driving_id,
                reference_upload_id=reference_id,
                generated_upload_id=generated_id,
                status=status,
                log=log
            )
        else:
            await report_assignment(
                fetcher=self.fetcher,
                assignment_id=assignment_id,
                driving_upload_id=None,
                reference_upload_id=None,
                generated_upload_id=None,
                status=status,
                log=log
            )

    async def run(self):
        if not session.is_authenticated():
            await self.authorize()

        while True:
            try:
                task = await self.claim_task()

                for p in list(task.values())[1::]:
                    if not Path(p).exists:
                        raise Exception(f"Cannot find saved image: {p}")

                path_map = self.inference(
                    face_path=task["driving_save_path"],
                    shape_path=task["reference_save_path"],
                    # Keep same color as original
                    color_path=task["driving_save_path"],
                    face_side_path=task["drving_side_save_path"],
                )

                for p in path_map.values():
                    if not Path(p).exists():
                        raise Exception(f"Cannot find generated image: {p}")

                await self.report(
                    path_map=path_map,
                    assignment_id=task["assignment_id"],
                    log="Sucessfully ran task",
                    status=AssignmentStatus.SUCCEED
                )

            # Map error (only unauthorized will be handled)
            except FrontError as e:
                if e.category == ErrorCategories.UNAUTHORIZED:
                    await self.authorize()

            except Exception as e:
                if task and task.get("assignment_id"):
                    self.report(
                        path_map=None,
                        assignment_id=task.get("assignment_id"),
                        log=str(e),
                        status=AssignmentStatus.FAILED
                    )
                raise


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str)
    parser.add_argument("--password", type=str)
    parser.add_argument("--base_url", type=str)

    args = parser.parse_args()

    worker = Worker(
        username=args.username,
        password=args.password,
        base_url=args.base_url,
    )

    asyncio.run(worker.run())
