from uuid import uuid4
from pathlib import Path
from manager.internal.image import insert_image
from manager.internal.task import create_task
from manager.schemas.image import ImageCategories
from manager.core.config import settings


def insert_tasks(cache_path: str):
    filename = Path(cache_path)

    if not filename.exists():
        print(f"File {filename} not found")
        return

    with open(filename, "r") as f:
        records = f.readlines()
        length = len(records)

        for i in range(length):
            print(f"Inserting {i} / {length} items")
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

            create_task(
                driving_id=drive_front_id,
                reference_id=ref_id,
                path=str(generated_path),
                priority=1,
                host="localhost"
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, default="cache.txt")
    args = parser.parse_args()
    insert_tasks(cache_path=args.cache)
