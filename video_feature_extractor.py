import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import time

class VideoFeatureExtractor():

    def __init__(self):
        # Init Pytorch pretrained model and preprocessing module
        self.model = models.mobilenet_v2(pretrained=True)
        # remove last fully-connected layer
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_features(self, video_path, start_time, end_time, N_sample_frames = 5):
        # if start_time == None or end_time == None:

        cap = cv2.VideoCapture(video_path)
        total_frame_N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # fps = cam.get(cv2.CAP_PROP_FPS)

        start_frame_id = int(start_time * fps)
        end_frame_id = int(end_time * fps)

        sampled_frames = []
        sample_frame_ids = np.linspace(start_frame_id, end_frame_id, N_sample_frames)
        sample_frame_ids = sample_frame_ids.astype(int)

        for i in sample_frame_ids:
            ret, frame = cap.read()
            # OpenCV read image in BGR order. we should transfer it to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames += [frame]
            cap.set(1, i)

        input_imgs_tensors = [self.preprocess(img) for img in sampled_frames]
        input_imgs_tensors = torch.stack(input_imgs_tensors)
        with torch.no_grad():
            out_features = self.model(input_imgs_tensors)  # 1280 * frames
            out_features = out_features.mean(axis=0)  # 1280
            return out_features

if __name__ == "__main__":
    video_feature_extractor = VideoFeatureExtractor()

    s_t = time.time()
    features = video_feature_extractor.get_features(video_path='./data/index.mp4')

    print(features)
    print("features.shape, type(features)", features.shape, type(features))
    print("Feature extract time for one video:", time.time() - s_t)
# plt.imshow(sampled_frames[1])


