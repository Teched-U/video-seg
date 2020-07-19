import os
import pickle
import json
import glob
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Tuple, List, Dict, Any
import random
from video_feature_extractor import VideoFeatureExtractor

from gensim.models.keyedvectors import KeyedVectors

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Max length of a sentecnce in a shot
MAX_LENGTH = 128
MIN_SHOTS_NUM = 16

GOOGLE_MODEL_PATH = "/media/word2vec/GoogleNews-vectors-negative300.bin"
STOPWORD_PATH = "data/stopwords_en.txt"


class DocSim(object):
    def __init__(self, w2v_model, stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc, seperate=False):
        """Identify the vector values for each word in the given document"""
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                # Ignore, if the word doesn't exist in the vocabulary
                pass

        if not word_vecs:
            if seperate:
                return [torch.zeros((300,))]
            else:
                return torch.zeros((300,))

        if seperate:
            vector = [torch.from_numpy(vec) for vec in word_vecs]
        else:
            # Assuming that document vector is the mean of all the word vectors
            # PS: There are other & better ways to do it.
            mean_vec = np.mean(word_vecs, axis=0)
            vector = torch.from_numpy(mean_vec)

        return vector


def init_word2vec(model_path: str, stopwords_file: str) -> Tuple[DocSim, List[str]]:
    with open(stopwords_file, "r") as f:
        stopwords = f.read().split(",")
        model = KeyedVectors.load_word2vec_format(
            model_path, binary=True, limit=1000000
        )
        docSim = DocSim(model, stopwords=stopwords)

        return docSim, stopwords


class VideoSegDataset(Dataset):
    # Adjust this field  when new features are added
    FEATURE_SIZE = 1584

    def __init__(
        self,
        data_folder,
        result_folder,
        seperate=True,
        save_dir=None,
        load_dir=None,
        num_frame=5,
    ):
        self.data_folder = data_folder

        # Get data into memory
        data_files = glob.glob(f"{data_folder}/*.json")

        # Init Word2Vec model
        self.docsim_model, self.stopwords = init_word2vec(
            GOOGLE_MODEL_PATH, STOPWORD_PATH
        )

        # Whether to seperate the features
        self.seperate = seperate

        # Get features and labels
        self.features = []

        # Timestamps for each video
        self.timestamps = []

        # Ground truth timestamps (processed) for each video
        self.gts = []
        self.raw_gts = []

        for data_file in data_files:
            # Get feature
            feature = None

            with open(data_file, "rb") as f:
                data = {}
                try:
                    data = pickle.load(f)
                except Exception:
                    print(f"error loading {data_file}")
                    continue

                if load_dir and os.path.exists(
                    os.path.join(load_dir, os.path.basename(data_file))
                ):
                    load_path = os.path.join(load_dir, os.path.basename(data_file))
                    feature = torch.load(load_path)
                else:
                    feature = self.encode_features(data, seperate, num_frame=num_frame)

                # Save for use later
                if save_dir:
                    save_path = os.path.join(save_dir, os.path.basename(data_file))
                    torch.save(feature, save_path)

                self.features.append(feature)
                timestamp = [shot["timestamp"] for shot in data["features"]]
                self.timestamps.append(timestamp)

            # Get label
            video_name = os.path.basename(data_file)
            gt_file = os.path.join(result_folder, f"gt_{video_name}")

            with open(gt_file, "r") as f:
                gt_data = json.load(f)
                self.raw_gts.append(gt_data)

                # one hot encode the ground truth to match the sequence of shots
                gt = self.encode_gt(gt_data, timestamp, video_name)
                self.gts.append(gt)

        self.num_samples = len(self.features)

        print(f"Dataset initialied: num_samples:{self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Return:
            featurs tensor: tensor[num_shot, D=300]
            gt tensor: tensor[num_shot, (0,1)]
        """
        # FIXME(XC): filter the feature before hand
        if len(self.features[idx]) < MIN_SHOTS_NUM:
            idx = 0 if idx else 1

        if self.seperate:
            return self.features[idx], self.gts[idx]

        return torch.stack(self.features[idx]), self.gts[idx]

    # FIXME(oyzh): check how features are extracted
    def encode_features(
        self, data: List[Dict], seperate=False, num_frame=5
    ) -> List[Any]:
        """
        Encode the features for one long video that consists of multiple shots.
        Args:
            data: information of the segments: 
                [{
                    "features": {
                        "duration": ...,
                        "pitch": ...,
                        "transcript": ...,
                        "timestamp": ...
                    },
                    "video_name": ...
                }]
            
            seperate: Flag to indicate whether features are combined
        """

        video_feat = self.encode_video_features(data, seperate, num_frame)
        audio_feat = self.encode_audio_features(data)
        asr_feat = self.encode_asr_features(data, seperate)

        if seperate:
            features = {"video": video_feat, "asr": asr_feat, "audio": audio_feat}
        else:
            features = [
                torch.cat(feat_tuple, dim=0)
                for feat_tuple in zip(video_feat, audio_feat, asr_feat)
            ]

        return features

    def encode_video_features(
        self, data: List[Dict], seperate=False, num_frame=5
    ) -> List[Any]:
        # Encode video features
        timestamps = [shot["timestamp"] for shot in data["features"]]
        video_path = data["video_name"]

        extractor = VideoFeatureExtractor()
        video_feature = extractor.get_features(
            video_path, timestamps, seperate=seperate, N_sample_frames=num_frame
        )

        return video_feature

    def encode_audio_features(self, data: List[Dict]) -> List[torch.Tensor]:
        audio_vec_arr = []
        for shot in data["features"]:
            audio_vec = torch.Tensor(
                [shot["duration"], shot["pitch"], shot["volume"], shot["pause"]]
            )

            audio_vec_arr.append(audio_vec)

        return audio_vec_arr

    def encode_asr_features(self, data: List[Dict], seperate=False) -> List[Any]:
        """
        If seperate, output a list of list of tensors, where the first list 
        contains list of feature vectors. (Not doing mean)
        If not seperate, output a list of tensors
        """
        sentences = [shot["transcript"] for shot in data["features"]]
        timestamps = [shot["timestamp"] for shot in data["features"]]
        video_path = data["video_name"]

        # Encode transcript features
        # 300 dim : feature vector
        result = [
            self.docsim_model.vectorize(sentence, seperate) for sentence in sentences
        ]
        return result

    def encode_gt(self, data, timestamp, video):
        """
        data: {
            "timestamp": "topic",
        }
        """

        gt = torch.zeros(len(timestamp), dtype=torch.long)
        TIMESTAMP_THRESHOLD = 5
        for gt_ts in data.keys():
            gt_ts = float(gt_ts)
            for idx, ts in enumerate(timestamp):
                if abs(gt_ts - ts) < TIMESTAMP_THRESHOLD:
                    gt[idx] = 1

        # # Naive of finding matches
        # gt_ts_dict = {}
        # for gt_ts in data.keys():
        #     gt_ts = float(gt_ts)
        #     # For each gt timestamp, find a matching segment (assuming there is)
        #     min_d = 100000 # no video longer than  this seconds
        #     min_idx = -1
        #     for idx, ts in enumerate(timestamp):
        #         if abs(gt_ts - ts) < min_d:
        #             min_d = abs(gt_ts -ts)
        #             min_idx = idx

        #     if min_idx == -1:
        #         print(f"issue with this {video}. some segmnet not assigned {gt_ts}")

        #     _, cur_min_d = gt_ts_dict.get(gt_ts, (0, 100000))

        #     if cur_min_d > min_d:
        #         gt_ts_dict[gt_ts] = (min_idx, min_d)
        #

        # for min_idx, min_d in gt_ts_dict.values():
        #     gt[min_idx] = 1
        # # Start of the video always a start
        # gt[0]  = 1

        return gt

    def shuffle(self):
        data = list(zip(self.features, self.gts, self.timestamps, self.raw_gts))
        random.shuffle(data)
        self.features, self.gts, self.timestamps, self.raw_gts = zip(*data)

    def get_ts(self, idx: int) -> List[float]:
        return self.timestamps[idx]

    def get_raw_gt(self, idx: int) -> List[float]:
        return self.raw_gts[idx]
