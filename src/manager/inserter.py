import os
from pathlib import Path
from manager.internal.image import insert_image
from manager.internal.task import create_task
from manager.schemas.image import ImageCategories


def insert_tasks():
    filename = Path("cache.txt")

    with open(filename, "r") as f:
        records = f.readlines()
        length = len(records)

        for i in range(length):
            print(f"Inserting {i} / {length} items")
            line = records[i]
            drive_front_path, drive_side_path, ref_path = line.strip().split(',')
            asset_dir = '/'.join(drive_front_path.split('/')[:-2])

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
                priority=1,
                host="localhost"
            )


if __name__ == "__main__":
    insert_tasks()
