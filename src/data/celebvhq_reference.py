import random
from pathlib import Path
from data.celebvhq_base import _CelebVHQBase


class CelebVHQReferenceDataset(_CelebVHQBase):
    """
    Optimized Dataset for:
      - driving_images/{id}_frontal.jpg
      - driving_images/{id}_side.jpg
      - reference_images/<random>.jpg

    Returns:
      {
          "front": Tensor[C,H,W],
          "side": Tensor[C,H,W],
          "reference": Tensor[C,H,W]
      }
    """

    def __init__(self, driving_dir: str, reference_dir: str, transform=None):
        super().__init__(driving_dir, transform)
        self.reference_dir = Path(reference_dir)
        self.samples = self._scan_driving_images()
        if len(self.samples) == 0:
            raise RuntimeError("No driving images found!")

        self.reference_paths = list(Path(reference_dir).glob("*.[jp][pn]g"))
        if len(self.reference_paths) == 0:
            raise RuntimeError("No reference images found!")

    def _scan_driving_images(self):
        entries = set()
        for p in self.driving_dir.iterdir():
            entries.add("_".join(p.name.split("_")[:-1]))

        return [{
            "id": id_,
            "front": self.driving_dir / f"{id_}_frontal.jpg",
            "side": self.driving_dir / f"{id_}_side.jpg",
        } for id_ in entries]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ref_path = random.choice(self.reference_paths)

        front = self._apply(self._load_image(sample["front"]))
        side = self._apply(self._load_image(sample["side"]))
        ref = self._apply(self._load_image(ref_path))

        return {
            "front": {"path": str(sample["front"]), "content": front},
            "side": {"path": str(sample["side"]), "content": side},
            "reference": {"path": str(ref_path), "content": ref},
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

    dataset = CelebVHQReferenceDataset(driving_dir="./assets/driving_images",
                                       reference_dir="./assets/reference_images",
                                       transform=transform)

    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=2,
                            pin_memory=True, drop_last=True)

    epoch_iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in epoch_iterator:
        print(batch.keys())
