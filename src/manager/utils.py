from tqdm import tqdm
from uuid import uuid4
from pathlib import Path
from datetime import datetime
from manager.internal.image import insert_image
from manager.internal.task import create_task
from manager.schemas.image import ImageCategories
from manager.core.config import settings
from manager.core.exceptions import AppError
from manager.internal.connect import get_cursor


def extract_meta(path):
    name = path.split('/')[-1]
    id = '_'.join(name.split('_')[:-1])
    ext = name.split('.')[-1]
    dir = '/'.join(path.split('/')[:-1])
    return {
        "name": name,
        "ext": ext,
        "dir": dir,
        "id": id
    }


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


def retrieve_completed_cache(cache_path: str):
    if cache_path.strip() == "auto":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = Path(f"{ts}.cache.txt")
    else:
        cache_path = Path(cache_path)

    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
            SELECT
                d.file_path AS driving_path,
                r.file_path AS reference_path
            FROM tasks t
            JOIN images d ON d.id = t.driving_image_id
            JOIN images r ON r.id = t.reference_image_id
            WHERE t.status = 'completed'::task_status
        """)
        tasks = cur.fetchall()

    if not tasks:
        print("No completed tasks found!")
        return

    with open(cache_path, "a") as f:
        for task in tqdm(tasks, total=len(tasks)):
            meta = extract_meta(task["driving_path"])
            side_path = f"{meta['dir']}/{meta['id']}_side.{meta['ext']}"
            f.write(
                f"{task['driving_path']},{side_path},{task['reference_path']}\n")


if __name__ == "__main__":
    import argparse
    DEFAULT_CACHE_PATH = Path(Path(__file__).parent, "assets/cache.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str,
                        help="Path to cache file",
                        default=DEFAULT_CACHE_PATH)
    parser.add_argument("--action", type=str,
                        help="Specify which action to perform",
                        required=True)
    args = parser.parse_args()

    ACTION_MAP = {
        "insert": lambda args: insert_tasks(cache_path=args.cache),
        "retrieve": lambda args: retrieve_completed_cache(cache_path=args.cache)
    }

    # Retreive and execute the action
    ACTION_MAP[args.action](args)
