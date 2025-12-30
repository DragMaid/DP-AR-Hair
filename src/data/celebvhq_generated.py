from pathlib import Path
from data.celebvhq_base import _CelebVHQBase


class CelebVHQGeneratedDataset(_CelebVHQBase):
    """
    Optimized Dataset for:
      - driving_images/{id}_frontal.jpg
      - driving_images/{id}_side.jpg
      - generated_images/{id}_generated.*

    Returns:
      {
          "front": Tensor[C,H,W],
          "side": Tensor[C,H,W],
          "reference": Tensor[C,H,W]
      }
    """

    def __init__(self, driving_dir: str, generated_dir: str, transform=None):
        super().__init__(driving_dir, transform)
        self.generated_dir = Path(generated_dir)
        self.samples = self._get_samples()
        if len(self.samples) == 0:
            raise RuntimeError("No sample images found!")

    def _get_samples(self):
        samples = []

        for p in self.generated_dir.iterdir():
            if (p.suffix.lower() not in [".jpg", ".jpeg", ".png"]):
                continue
            filename = "_".join(p.name.split("_")[:-1])
            generated_path = Path.joinpath(
                self.generated_dir, filename + f"_generated{p.suffix}")
            front_path = Path.joinpath(
                self.driving_dir, filename + "_frontal.jpg")
            side_path = Path.joinpath(
                self.driving_dir, filename + "_side.jpg")

            if (front_path.exists() and side_path.exists()):
                samples.append({
                    "id": filename,
                    "front": front_path,
                    "side": side_path,
                    "generated": generated_path
                })

        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        front = self._apply(self._load_image(sample["front"]))
        side = self._apply(self._load_image(sample["side"]))
        generated = self._apply(self._load_image(sample["generated"]))

        return {
            "front": {"path": str(sample["front"]), "content": front},
            "side": {"path": str(sample["side"]), "content": side},
            "reference": {"path": str(sample["generated"]), "content": generated},
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

    dataset = CelebVHQGeneratedDataset(driving_dir="./assets/driving_images",
                                       generated_dir="./assets/generated_images",
                                       transform=transform)

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True)

    epoch_iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in epoch_iterator:
        print(batch.keys())
