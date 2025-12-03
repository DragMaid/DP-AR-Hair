import torch
from pipelines.training_pipeline import TrainingPipeline
from torchvision import transforms as T
from PIL import Image
import os


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transforms
    # transform = T.Compose([
        # T.Resize((256, 256)),   # adjust to your network's expected size
        # T.ToTensor(),           # yields CHW float in [0,1]
    # ])

    # Initialize pipeline
    pipeline = TrainingPipeline(device=device)
    scaler = torch.cuda.amp.GradScaler()

    # Load 3 images
    img_paths = [
        "./assets/test_images/phuc.jpeg",
        "./assets/test_images/phuc.jpeg",
        "./assets/test_images/ken.png"
    ]
    images = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        # img = transform(img)
        # img = img.unsqueeze(0).to(device)  # add batch dim
        images.append(img)

    # Assign to source, driving, reference
    I_s, I_d, I_r = images

    # Run a train step
    logs = pipeline.train_step(I_s, I_d, I_r, scaler=scaler)
    print("Train step logs:", logs)

    # Save checkpoint
    ck_path = os.path.join(dir, f"epoch_{1:04d}.pt")
    pipeline.save_checkpoint(ck_path, epoch=1)
    print(f"Checkpoint saved to {ck_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
