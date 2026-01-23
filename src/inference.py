import cv2
import argparse
import torch
import math
from torchvision import transforms as T
from pipelines.inference_pipeline import InferencePipeline
from configs.pipeline_config import pipeline_config as pco
from sixdrepnet import SixDRepNet
from loaders.loader import load_hfg_generator
from hair_gan.utils.shape_predictor import align_face


class HairShifter:
    def __init__(self, device, batch_size):
        self.device = device
        # The higher the batch the faster it processes
        self.batch_size = batch_size

        self.pipeline = InferencePipeline(device)
        self.generator = load_hfg_generator()
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
            timestamp = length * index
            video.set(cv2.CAP_PROP_POS_FRAMES, timestamp)
            res, frame = video.read()
            pitch, yaw, _ = self.pose_estimator(frame)
            samples.append([timestamp, pitch, yaw])

        ref_img = cv2.imread(reference_path)
        r_pitch, r_yaw, _ = self.pose_estimator(ref_img)

        difference_min = [0, math.inf]
        for sample in samples:
            difference = abs(sample[1] - pitch) + abs(sample[2] - yaw)
            if difference < difference_min:
                difference_min = [sample[0], difference]

        video.set(cv2.CAP_PROP_POS_FRAMES, difference_min[0])
        _, frame = video.read()
        return frame

    def transfer(self, video_path, referece_path, output_path):
        video = cv2.VideoCapture(video_path)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # TODO: check for size consistency
        # TODO: move this to a config file
        fps = 30
        frame_size = (512, 512)
        out_video = cv2.VideoWriter(
            filename='output.mp4',
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=fps,
            frameSize=frame_size
        )

        anchor_frame = self.select_driving(video_path, referece_path)
        anchor_frame = align_face(anchor_frame, predictor=self.predictor)[0]

        ref_img = cv2.imread(referece_path)
        ref_img = align_face(ref_img, predictor=self.predictor)[0]

        source_frame = self.generator(
            face_img=anchor_frame,
            shape_img=ref_img,
            color_img=ref_img,
            align=False  # All aligned already
        )

        driving_tensors = []
        count = 0

        while True:
            success, frame = video.read()
            if not success:
                break

            driving_tensors.append(frame)
            count += 1

            is_full_batch = len(driving_tensors) == self.batch_size
            is_last_batch = count == length

            if is_full_batch or is_last_batch:
                driving_batch = torch.stack(driving_tensors)

                source_tensor = self.transform(source_frame)
                source_batch = torch.stack(
                    [source_tensor for _ in range(driving_batch.shape[0])])

                driving_tensors.clear()

                output_batch = self.pipeline.inference(
                    I_s=source_batch,
                    I_d_t=driving_batch
                )

                for i in range(output_batch.shape[0]):
                    frame = output_batch[i]
                    frame = frame.permute(1, 2, 0)
                    frame = frame.clamp(0, 1)
                    frame = (frame * 255).byte()
                    frame = frame.cpu().numpy()

                    frame = frame[:, :, ::-1]
                    out_video.write(frame)

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
                   default=pco.inference.reference_path)
    p.add_argument("--batch_size", type=str, help="Batch size to pass into model",
                   default=pco.inference.batch_size)
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    shifter = HairShifter(
        device=device,
        batch_size=args.batch_size
    )

    shifter.tranfer(
        video_path=args.video,
        reference_path=args.reference,
        output_path=args.output,
    )

    print("Inference complete.")


if __name__ == "__main__":
    main()
