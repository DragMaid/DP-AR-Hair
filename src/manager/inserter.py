import os
from pathlib import Path
from internal.image import insert_image
from internal.task import create_task
from schemas.image import ImageTypes


def insert_tasks():
    filename = Path("cache.txt")

    with open(filename, "r") as f:
        for line in f.readlines():
            drive_front_path, drive_side_path, ref_path = line.strip().split(',')
            asset_dir = '/'.join(drive_front_path.split('/')[:-2])

            drive_front_id = insert_image(drive_front_path, ImageTypes.DRIVING)
            _ = insert_image(drive_side_path, ImageTypes.DRIVING)
            ref_id = insert_image(ref_path, ImageTypes.REFERENCE)

            drive_front_name = drive_front_path.split('/')[-1]
            # the id might also include special characters like '_' (Ex: 12_d_frontal.jpg)
            drive_front_ori_id = '_'.join(drive_front_name.split('_')[:-1])

            generated_path = os.path.join(
                asset_dir,
                "generated_images/",
                f"{drive_front_ori_id}_generated.jpg"
            )

            create_task(
                driving_id=drive_front_id,
                reference_id=ref_id,
                path=generated_path,
                priority=1
            )


if __name__ == "__main__":
    insert_tasks()
