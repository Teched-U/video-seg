# Ref: https://github.com/marl/openl3
# paper: http://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf
import openl3
import soundfile as sf
from moviepy.editor import *
import numpy as np
import time
import tensorflow as tf


class AudioFeatureExtractor():

    def __init__(self):
        return

    # There might be a bug if the length of audio file is too short.
    # We sampled 5 pieces of 512-length audio samples from a video file, and extract a 512 dim feature
    def get_features(self, video_path, timestamps, N_sample_frames = 5, sample_Len = None):
        # Read from audio file
        # audio_file_path = '/Users/ouyangzhihao/PycharmProjects/TechedU/explore/dev-pyannote-audio/tests/data/trn00.wav'
        # audio, sr = sf.read(audio_file_path)

        # Read audio from video file
        audio = VideoFileClip(video_path).audio
        audio, sr = audio.to_soundarray(), audio.fps # sr means sample rate of audio file

        total_frame_N = len(audio)
        # audio_list = []
        # if not sample_Len: sample_Len = sr
        # sample_frame_ids = np.linspace(0, total_frame_N-sample_Len, N_sample_frames)
        # sample_frame_ids = sample_frame_ids.astype(int)
        # for id in sample_frame_ids:
        #     audio_list += [audio[id:id+sample_Len]]

        start_time = time.time()
        emb_list, ts_list = openl3.get_audio_embedding(audio, sr, hop_size=1.0, batch_size=8, embedding_size=512)
        print("Feature extraction time:", time.time() - start_time)
        # emb_list, ts_list = openl3.get_audio_embedding(audio_list, sr, batch_size=32, embedding_size=6144)
        print("Original embedding: ", np.shape(emb_list), np.shape(ts_list))
        small_idx, large_idx = 0, 0
        ret_embs = []
        for i in len(1, timestamps):
            label_t_start, label_t_end = timestamps[i - 1], timestamps[i]
            emb_start_idx, emb_end_idx = 0, 1
            start_dis, end_dis = 999999999999999, 9999999999999999
            for emb_idx, emb_t in enumerate(ts_list):
                if start_dis > abs(emb_t-label_t_start):
                    start_dis = abs(emb_t-label_t_start)
                    emb_start_idx = emb_idx
                if end_dis > abs(emb_t-label_t_end):
                    end_dis = abs(emb_t-label_t_end)
                    emb_end_idx = emb_idx
            if emb_start_idx == emb_end_idx:
                ret_embs += [[emb_list[emb_idx]]]
            else:
                ret_embs += [[emb_list[emb_start_idx:emb_end_idx].mean()]]
        return ret_embs





        # import ipdb; ipdb.set_trace()

        # emb_mean = np.array(emb_list).mean(axis=0) # transfer list to numpy array and get the mean as the feature
        # emb_mean = emb_mean.mean(axis=0) # Mean the "default intervals (0.1s)" of the audio file
        # emb_mean = emb_mean.squeeze() # flatten the feature

        return emb_mean

    def get_features_slow(self, video_path, N_sample_frames = 5):
        # Read from audio file
        # audio_file_path = '/Users/ouyangzhihao/PycharmProjects/TechedU/explore/dev-pyannote-audio/tests/data/trn00.wav'
        # audio, sr = sf.read(audio_file_path)

        # Read audio from video file
        audio = VideoFileClip(video_path).audio
        audio, sr = audio.to_soundarray(), audio.fps # sr means sample rate of audio file
        hop_size_seconds = (len(audio) // N_sample_frames) / sr
        print('hop_size_seconds', hop_size_seconds)
        emb, ts = openl3.get_audio_embedding(audio, sr, hop_size=hop_size_seconds, embedding_size=512)

        # import ipdb; ipdb.set_trace()
        emb_mean = emb.mean(axis=0) # transfer list to numpy array and get the mean as the feature
        emb_mean = emb_mean.squeeze() # flatten the feature

        return emb_mean

if __name__ == "__main__":
    video_feature_extractor = AudioFeatureExtractor()
    s_t = time.time()
    features = video_feature_extractor.get_features(video_path='./data/index.mp4') # Faster but may have bugs
    # features = video_feature_extractor.get_features_slow(video_path='./data/index.mp4') # More stable but slower

    print(features)
    print("features.shape, type(features)", features.shape, type(features))
    print("Feature extract time for one video:", time.time() - s_t)
