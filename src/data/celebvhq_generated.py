from pathlib import Path
from data.celebvhq_base import _CelebVHQBase


class CelebVHQGeneratedDataset(_CelebVHQBase):
    """
    Optimized Dataset for:
      - {id}_frontal.jpg
      - {id}_side.jpg
      - {id}_generated.*

    Returns:
      {
          "front": Tensor[C,H,W],
          "side": Tensor[C,H,W],
          "reference": Tensor[C,H,W]
      }
    """

    def __init__(self, dataset_dir: str, transform=None):
        super().__init__(dataset_dir, transform)
        self.dataset_dir = Path(dataset_dir)
        self.samples = self._get_samples()
        if len(self.samples) == 0:
            raise RuntimeError("No sample images found!")

    def _get_samples(self):
        samples = []

        # print(list(self.dataset_dir.iterdir()))
        for p in self.dataset_dir.glob("*_driving.*"):

            if (p.suffix.lower() not in [".jpg", ".png"]):
                continue

            filename = "_".join(p.name.split("_")[:-1])
            generated_path = Path.joinpath(
                self.dataset_dir, filename + f"_generated{p.suffix}")
            driving_path = Path.joinpath(
                self.dataset_dir, filename + f"_driving{p.suffix}")
            reference_path = Path.joinpath(
                self.dataset_dir, filename + f"_reference{p.suffix}")

            if (generated_path.exists() and driving_path.exists() and reference_path.exists()):
                samples.append({
                    "id": filename,
                    "driving": driving_path,
                    "reference": reference_path,
                    "generated": generated_path
                })

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        driving = self._apply(self._load_image(sample["driving"]))
        reference = self._apply(self._load_image(sample["reference"]))
        generated = self._apply(self._load_image(sample["generated"]))

        return {
            "driving": {"path": str(sample["driving"]), "content": driving},
            "reference": {"path": str(sample["reference"]), "content": reference},
            "generated": {"path": str(sample["generated"]), "content": generated},
        }


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torchvision import transforms as T

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    dataset = CelebVHQGeneratedDataset(dataset_dir="./src/manager/assets/generated/",
                                       transform=transform)

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True)

    epoch_iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in epoch_iterator:
        print(batch.keys())
