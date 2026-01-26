import PIL
import cv2
import argparse
import torch
import math
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms as T
from pipelines.inference_pipeline import InferencePipeline
from configs.pipeline_config import pipeline_config as pco
from sixdrepnet import SixDRepNet
from loaders.loader import load_hfg_generator
from hair_gan.utils.shape_predictor import align_face, get_landmark_detector


def cv2_to_pil(cv2_image):
    coverted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(coverted)
    return pil_image


def cv2_to_tensor(frame):
    # frame: uint8 BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame /= 255.0
    return frame


class HairShifter:
    def __init__(self, device, batch_size, checkpoint):
        self.device = device
        # The higher the batch the faster it processes
        self.batch_size = batch_size

        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint} not found")

        self.pipeline = InferencePipeline(device)
        self.pipeline.load_checkpoint(checkpoint)

        self.generator = load_hfg_generator()
        self.predictor = get_landmark_detector()
        self.pose_estimator = SixDRepNet(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def select_anchor(self, video_path, reference_path, stride=10):
        video = cv2.VideoCapture(video_path)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        samples = []

        for index in range(stride):
            timestamp = index * (length // stride)
            video.set(cv2.CAP_PROP_POS_FRAMES, timestamp)
            res, frame = video.read()
            if not res:
                raise ValueError("Corrupted or unsupported video format")
            pitch, yaw, _ = self.pose_estimator.predict(frame)
            samples.append([timestamp, pitch, yaw])

        ref_img = cv2.imread(reference_path)
        r_pitch, r_yaw, _ = self.pose_estimator.predict(ref_img)

        difference_min = [0, math.inf]
        for sample in samples:
            difference = abs(sample[1] - r_pitch) + abs(sample[2] - r_yaw)
            if difference < difference_min[1]:
                difference_min = [sample[0], difference]

        video.set(cv2.CAP_PROP_POS_FRAMES, difference_min[0])
        res, frame = video.read()
        if not res:
            raise ValueError("Unable to get final anchor frame")
        video.release()
        return frame

    def transfer(self, video_path, reference_path, output_path):
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video path {video_path} not found")

        if not Path(reference_path).exists():
            raise FileNotFoundError(
                f"Reference path {reference_path} not found")

        video = cv2.VideoCapture(video_path)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # TODO: check for size consistency
        # TODO: move this to a config file
        fps = 30
        frame_size = (256, 256)
        out_video = cv2.VideoWriter(
            filename=output_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=fps,
            frameSize=frame_size
        )

        anchor_frame = self.select_anchor(video_path, reference_path)
        anchor_frame = cv2_to_pil(anchor_frame)

        ref_img = cv2.imread(reference_path)
        ref_img = cv2_to_pil(ref_img)

        anchor_frame = align_face(anchor_frame, predictor=self.predictor)[0]
        ref_img = align_face(ref_img, predictor=self.predictor)[0]

        source_frame = self.generator(
            face_img=anchor_frame,
            shape_img=ref_img,
            color_img=ref_img,
            align=False  # All aligned already
        )

        # TODO: add poor mode
        del self.generator, self.pose_estimator

        driving_frames = []

        source_tensor = self.transform(source_frame).to(self.device)
        source_tensor = source_tensor.unsqueeze(0)  # [1, C, H, W]

        batch_buffer = torch.empty(
            (self.batch_size, 3, 256, 256),
            device=self.device
        )

        for i in tqdm(range(1, length+1)):
            success, frame = video.read()
            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # TODO: try removing the align face to see if it converged
            # TODO: if using align face then please do resize to (1024, 1024) also
            # TODO: see if there's any way to reduce the amount of transformation
            # TODO: see what is setting the output size
            batch_buffer[len(driving_frames)] = self.transform(frame)
            driving_frames.append(1)  # Just to count

            is_full_batch = len(driving_frames) == self.batch_size
            is_last_batch = i == length

            if is_full_batch or is_last_batch:
                driving_batch = batch_buffer[:len(driving_frames)]

                bs = driving_batch.shape[0]
                source_batch = source_tensor.expand(bs, -1, -1, -1)

                driving_frames.clear()

                with torch.inference_mode():
                    output_batch = self.pipeline.inference(
                        I_s=source_batch,
                        I_d_t=driving_batch
                    )

                del driving_batch, source_batch

                for i in range(output_batch.shape[0]):
                    frame = output_batch[i].detach().cpu()
                    frame = frame.permute(1, 2, 0)
                    frame = frame.clamp(0, 1)
                    frame = (frame * 255).byte().numpy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out_video.write(frame)

        out_video.release()
        print(f"Video created at: {output_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, help="Path to checkpoint",
                   default=pco.inference.checkpoint_path)
    p.add_argument("--video", type=str, help="Path to input video",
                   default=pco.inference.video_path)
    p.add_argument("--reference", type=str, help="Path to reference image",
                   default=pco.inference.reference_path)
    p.add_argument("--output", type=str, help="Path to output video",
                   default=pco.inference.output_path)
    p.add_argument("--batch_size", type=int, help="Batch size to pass into model",
                   default=pco.inference.batch_size)
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    shifter = HairShifter(
        device=device,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint
    )

    shifter.transfer(
        video_path=args.video,
        reference_path=args.reference,
        output_path=args.output,
    )

    print("Inference complete.")


if __name__ == "__main__":
    main()
