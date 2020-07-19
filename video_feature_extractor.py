import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

import tempfile
import time
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def get_length(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return float(result.stdout)


class VideoFeatureExtractor:
    def __init__(self):
        # Init Pytorch pretrained model and preprocessing module
        self.model = models.mobilenet_v2(pretrained=True).eval()
        # remove last fully-connected layer
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-1]
        )
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # # NOTE:This method below is too slow....
    # def process(self, N_sample_frames = 5):
    #     """
    #     For each video, it cuts the entire video into pieces first
    #     (according to its corresponding timestamps)
    #     """

    #     # Extract video shots
    #     len_video = get_length(self.video_path)
    #     self.timestamp_arr.append(len_video)
    #     i = 0
    #     feat_arr = []
    #     for i in range(len(self.timestamp_arr)-1):
    #         fd, clip_file = tempfile.mkstemp(suffix='.mp4')
    #         start_t = self.timestamp_arr[i]
    #         end_t = self.timestamp_arr[i+1]

    #         ffmpeg_extract_subclip(self.video_path, start_t, end_t, targetname=clip_file)
    #         print(clip_file)
    #         feat_arr.append(self.get_features(clip_file, N_sample_frames))

    #     return feat_arr

    def get_features(self, video_path, timestamps, N_sample_frames=5, seperate=False):
        cap = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frame_N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_features_arr = []

        for i in range(len(timestamps)):
            start = (
                int(timestamps[i] * fps)
                if int(timestamps[i] * fps) < total_frame_N
                else total_frame_N - 1
            )
            end = (
                int(fps * timestamps[i + 1])
                if i < len(timestamps) - 1
                else total_frame_N
            )

            sample_frame_ids = np.linspace(start, end, N_sample_frames)

            sampled_frames = []
            sample_frame_ids = sample_frame_ids.astype(int)

            for j in sample_frame_ids:
                ret, frame = cap.read()
                if ret:
                    # OpenCV read image in BGR order. we should transfer it to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sampled_frames += [frame]
                cap.set(1, j)

            input_imgs_tensors = [self.preprocess(img) for img in sampled_frames]
            input_imgs_tensors = torch.stack(input_imgs_tensors)

            with torch.no_grad():
                out_features = self.model(input_imgs_tensors)  # 1280 * frames

            if seperate:
                # Have the sequence feature extractor to do the job
                feat = out_features
            else:
                feat = out_features.mean(axis=0)  # 1280

            out_features_arr.append(feat)

        return out_features_arr


if __name__ == "__main__":
    video_feature_extractor = VideoFeatureExtractor()

    s_t = time.time()
    features = video_feature_extractor.get_features(video_path="./data/index.mp4")

    print(features)
    print("features.shape, type(features)", features.shape, type(features))
    print("Feature extract time for one video:", time.time() - s_t)
# plt.imshow(sampled_frames[1])
