import pickle
from test import VideoSegPrediction


def run_model(video_file, input_file, feature_dir):
    video_model = VideoSegPrediction('/data/Exp/test-dev-video-seg/checkpoint/best_model.pd')
    res = video_model.test_one_video(video_file, input_file, feature_dir)

    res = res.tolist()
    return res

