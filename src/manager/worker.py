import time
import requests
import sys
import traceback
from pathlib import Path
from PIL import Image
from urllib.parse import urljoin
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from loaders.loader import load_hfg_generator

from httpx import AsyncClient
from manager.client.core.session import session
from manager.client.api.fetcher import APIFetcher
from manager.client.core.config import settings
from manager.client.api.auth import authorize
from manager.schemas.user import UserRoles
from manager.schemas.image import ImageCategories
from manager.schemas.assignment import AssignmentStatus
from manager.client.api.image import upload
from manager.client.api.assignment import report_assignment
from manager.client.core.errors import FrontError, ErrorCategories
from manager.typings.backend import ClaimTaskResponse

from hair_gan.utils.shape_predictor import get_landmark_detector, align_face

SAVE_DIR = "./assets/results/"
TMP_DIR = "./assets/tmp/"
MAX_AUTH_COUNT = 5


class Worker:

    def __init__(
        self,
        username: str,
        password: str,
        base_url: Optional[str] = None
    ):
        self.username = username
        self.password = password
        self.base_url = base_url if base_url else settings.BASE_URL

        self.client = AsyncClient()
        self.fetcher = APIFetcher(
            base_url=urljoin(self.base_url, '/api'),
            client=self.client,
            session=session,
            strict=False
        )

        self.task = None
        self._auth_failed_count = 0

        self._last_claim_ts = 0.0
        self._min_claim_interval = 3

    def init_generator(self):
        """Load the hair fast gan generator and make it available."""
        self.model = load_hfg_generator()
        # The predictor is provided by HFG so please do only use it after init
        self.predictor = get_landmark_detector()

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

        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img.clamp(0, 1))
        img.save(path)
        print(f"[INFO] Saved output to {path}", file=sys.stderr)

    def download_image(self, source_path: str, output_path: str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        url = urljoin(self.base_url, source_path)
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
            map(self.convert_input, (
                face_path,
                shape_path,
                color_path,
                face_side_path
            ))
        )

        if not all(converted_inputs):
            print("[ERROR] Failed to load input images.", file=sys.stderr)
            raise

        face_obj, shape_obj, color_obj, face_side_obj = converted_inputs

        # Always perform alignment
        result = self.model(
            face_img=face_obj,
            shape_img=shape_obj,
            color_img=color_obj,
            predictor=self.predictor,
            align=True
        )

        side_aligned = align_face(face_side_obj, predictor=self.predictor)[0]

        generated_save_path = Path(SAVE_DIR, "generated.jpg")
        driving_save_path = Path(SAVE_DIR, "driving.jpg")
        reference_save_path = Path(SAVE_DIR, "reference.jpg")

        self.save_output(result["final_image"], generated_save_path)
        self.save_output(result["aligned_face"], driving_save_path)
        self.save_output(side_aligned, reference_save_path)

        return {
            "generated_path": generated_save_path,
            "driving_path": driving_save_path,
            "reference_path": reference_save_path
        }

    def get_driving_side_path(self, drive_path):
        drive_name = drive_path.split('/')[-1]
        drive_dir = '/'.join(drive_path.split('/')[:-1])
        drive_id = '_'.join(drive_name.split('_')[:-1])
        drive_ext = drive_name.split('.')[-1]
        return f"{drive_dir}/{drive_id}_side.{drive_ext}"

    async def authorize(self):
        try:
            await authorize(
                fetcher=self.fetcher,
                username=self.username,
                password=self.password,
                role=UserRoles.WORKER
            )
            self.auth_failed_count = 0
        except Exception as e:
            print(f"Authorization error: {e}")
            # If this fail then end it
            exit(1)

    async def handle_unauthorized(self, callback):
        if self._auth_failed_count >= MAX_AUTH_COUNT:
            exit(1)

        await self.authorize()
        self._auth_failed_count += 1
        await callback()

    async def _rate_limited_claim(self):
        now = time.time()
        elapsed = now - self._last_claim_ts

        if elapsed < self._min_claim_interval:
            await asyncio.sleep(self._min_claim_interval - elapsed)

        self._last_claim_ts = time.time()
        return await self.claim_task()

    async def claim_task(self):
        try:
            response = await self.fetcher.fetch(
                method="POST",
                path="/tasks/claim",
                require_auth=True,
                response_model=ClaimTaskResponse,
            )

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
        except FrontError as e:
            if e.category == ErrorCategories.UNAUTHORIZED:
                await self.handle_unauthorized(self.claim_task)
            else:
                raise

    async def report(self, path_map, assignment_id, log, status):
        try:
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
        except FrontError as e:
            if e.category == ErrorCategories.UNAUTHORIZED:
                await self.handle_unauthorized(
                    callback=lambda _: self.report(
                        path_map, assignment_id, log, status)
                )
            else:
                raise

    async def process(self):
        self.task = await self._rate_limited_claim()

        if not self.task:
            return

        input_paths = [
            self.task["driving_save_path"],
            self.task["driving_side_save_path"],
            self.task["reference_save_path"],
        ]

        for p in input_paths:
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Missing input image: {p}")

        path_map = await asyncio.to_thread(
            self.inference,
            face_path=self.task["driving_save_path"],
            shape_path=self.task["reference_save_path"],
            color_path=self.task["driving_save_path"],
            face_side_path=self.task["driving_side_save_path"],
        )

        for p in path_map.values():
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Missing generated image: {p}")

        await self.report(
            path_map=path_map,
            assignment_id=self.task["assignment_id"],
            log="Successfully ran task",
            status=AssignmentStatus.SUCCEED,
        )

        print("[WORKER] Finished task")

    async def report_failed(self, error: Exception):
        if self.task and self.task.get("assignment_id"):
            await self.report(
                path_map=None,
                assignment_id=self.task["assignment_id"],
                log=str(error),
                status=AssignmentStatus.FAILED,
            )

    async def run(self):
        print("[WORKER] Started")

        while True:
            start = time.time()
            self.task = None

            try:
                await self.process()

            except FrontError as e:
                print(f"[ERROR] FrontError: {e}")
                if e.category == ErrorCategories.QUEUE_EMPTY:
                    break
                await self.report_failed(e)

            except Exception as e:
                print("[ERROR] Task failed:")
                traceback.print_exc()
                await self.report_failed(e)

            finally:
                duration = time.time() - start
                print(f"[WORKER] Task duration: {duration:.2f}s")
                self.task = None


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str)
    parser.add_argument("--password", type=str)
    parser.add_argument("--base_url", type=str)

    args = parser.parse_args()

    async def run():
        worker = Worker(
            username=args.username,
            password=args.password,
            base_url=args.base_url,
        )
        await worker.authorize()
        worker.init_generator()
        await worker.run()

    asyncio.run(run())
