from tqdm import tqdm
from uuid import uuid4
from pathlib import Path
from typing import Union
from manager.internal.image import insert_image
from manager.internal.task import create_task
from manager.schemas.image import ImageCategories
from manager.core.config import settings
from manager.core.exceptions import AppError


def replace_parent(parent_dir: str, path: str):
    return str(Path(parent_dir, path.split('/')[-1]))


def insert_tasks(cache_path: str):
    filename = Path(cache_path)

    if not filename.exists():
        print(f"File {filename} not found")
        return

    with open(filename, "r") as f:
        records = f.readlines()
        length = len(records)

        count = 0
        for i in tqdm(range(length)):
            line = records[i]
            drive_front_path, drive_side_path, ref_path = line.strip().split(',')

            drive_front_id = insert_image(
                drive_front_path,
                ImageCategories.DRIVING,
                host="localhost"
            )
            _ = insert_image(
                drive_side_path,
                ImageCategories.DRIVING,
                host="localhost"
            )
            ref_id = insert_image(
                ref_path,
                ImageCategories.REFERENCE,
                host="localhost"
            )

            generated_name = f"{uuid4()}_generated.jpg"
            generated_path = Path(settings.GENERATED_IMAGE_DIR, generated_name)

            try:
                create_task(
                    driving_id=drive_front_id,
                    reference_id=ref_id,
                    path=str(generated_path),
                    priority=1,
                    host="localhost"
                )
                count += 1
            except AppError as e:
                if e.code != "TASK_CREATION_FAILED":
                    raise

        print(f"Process inserted {count} new tasks!")


if __name__ == "__main__":
    import argparse
    DEFAULT_CACHE_PATH = Path(Path(__file__).parent, "assets/cache.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Union[str, Path],
                        help="Path to cache file",
                        default=DEFAULT_CACHE_PATH)
    args = parser.parse_args()
    insert_tasks(cache_path=args.cache)
