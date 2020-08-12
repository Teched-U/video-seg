import os
import pickle
import json
import glob
from torch.utils.data import Dataset
import torch
import numpy as np
from typing import Tuple, List
import random

from gensim.models.keyedvectors import KeyedVectors
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
from transformers import BertTokenizer, BertModel
import torch
import time
from video_feature_extractor import *

# Max length of a sentecnce in a shot
MAX_LENGTH = 128
MIN_SHOTS_NUM = 16

GOOGLE_MODEL_PATH = '/media/word2vec/GoogleNews-vectors-negative300.bin'
STOPWORD_PATH = 'data/stopwords_en.txt'


class DocSim(object):
    def __init__(self, w2v_model, stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc):
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

        # Assuming that document vector is the mean of all the word vectors
        # PS: There are other & better ways to do it.
        if not word_vecs:
            return False, np.zeros((300,))

        vector = np.mean(word_vecs, axis=0)

        vector = torch.from_numpy(vector)

        return True, vector


def init_word2vec(model_path: str, stopwords_file: str) -> Tuple[DocSim, List[str]]:
    with open(stopwords_file, 'r') as f:
        stopwords = f.read().split(",")
        model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=1000000)
        docSim = DocSim(model, stopwords=stopwords)

        return docSim, stopwords


class VideoSegDataset(Dataset):
    def __init__(self, data_folder, result_folder):
        self.data_folder = data_folder

        # Get data into memory
        data_files = glob.glob(f"{data_folder}/*.json")
        print(data_folder)
        print("Total video transcript len(data_files)", len(data_files))

        # Init bert
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Video feature extractor
        self.video_feature_extractor = VideoFeatureExtractor()

        # Init Word2Vec model
        self.docsim_model, self.stopwords = init_word2vec(GOOGLE_MODEL_PATH, STOPWORD_PATH)

        # Get features and labels
        self.features = []

        # Timestamps for each video
        self.timestamps = []

        # video features
        self.video_fearures = []

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
                feature, timestamp, video_feature = self.encode_features(data)
                if feature:
                    self.features.append(feature)
                    self.timestamps.append(timestamp)
                    self.video_fearures.append(video_feature)
                else:
                    # Invalid
                    continue

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

        self.data = zip(self.features, self.gts)
        print(f"Dataset initialied: num_samples:{self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Return:
            featurs tensor: tensor[num_shot, D=300]
            gt tensor: tensor[num_shot, (0,1)]
        """
        # import ipdb;
        # ipdb.set_trace()
        feature_tensor = torch.stack(self.features[idx])
        video_tensor = self.video_fearures[idx]
        # video_tensor = torch.stack(self.video_fearures[idx])
        return torch.cat([feature_tensor, video_tensor], dim=1), self.gts[idx]
        # import ipdb; ipdb.set_trace()
        # print("__getitem__: Debug")
        # print("self.features[idx].shape", self.features[idx].shape)
        # print("self.video_fearures[idx].shape", self.video_fearures[idx].shape)
        # return torch.stack(
        #     torch.cat(self.features[idx], self.video_fearures[idx])
        # ), self.gts[idx]

    def run_bert(self, input_str):

        with torch.no_grad():
            inputs = self.bert_tokenizer.tokenize(input_str)
            if len(inputs) > 510: inputs = inputs[:510]

            inputs = [self.bert_tokenizer.cls_token] + inputs + [self.bert_tokenizer.sep_token]
            inputs = self.bert_tokenizer.convert_tokens_to_ids(inputs)
            inputs = torch.tensor(inputs, dtype=torch.long)
            inputs = torch.unsqueeze(inputs, 0)

            # self.bert_model(inputs.squeeze())
            outputs = self.bert_model(inputs)
            last_hidden_states = outputs[0]

            return last_hidden_states[0][0]

        # with torch.no_grad():
        #     # import ipdb;
        #     # ipdb.set_trace()
        #     input_str2 = input_str
        #     inputs2 = self.bert_tokenizer.tokenize(input_str2)
        #     inputs = self.bert_tokenizer.tokenize(input_str)
        #     inputs = [self.bert_tokenizer.cls_token] + inputs + [self.bert_tokenizer.sep_token] + inputs2 + [self.bert_tokenizer.sep_token]
        #     inputs = self.bert_tokenizer.convert_tokens_to_ids(inputs)
        #     inputs = torch.tensor(inputs, dtype=torch.long)
        #     inputs = torch.unsqueeze(inputs, 0)
        #
        #     # self.bert_model(inputs.squeeze())
        #     outputs = self.bert_model(inputs, token_type_ids=XX)
        #     last_hidden_states = outputs[0]
        #
        #     return last_hidden_states[0][0]

    def encode_features(self, data):
        sentences = [shot["transcript"] for shot in data['features']]
        timestamps = [shot["timestamp"] for shot in data['features']]

        # Run Bert
        print(f"Run bert for {data['video_name']}")
        video_folder = os.path.dirname(data['video_name'])
        bert_feature_path = os.path.join(video_folder, 'bert_feature.pkl')
        if not os.path.exists(bert_feature_path):
            s_t = time.time()
            result = [self.run_bert(sentence) for sentence in sentences]
            print("Time cose:", time.time() - s_t)
            torch.save(torch.stack(result), bert_feature_path)
        else:
            # print("Read bert feature", bert_feature_path)
            result = torch.load(bert_feature_path)

        vecs = []
        ts = []
        for res, t in zip(result, timestamps):
            vecs.append(res)
            ts.append(t)


        # TODO(OY): encode video features
        print(f"Run Video for {data['video_name']}")
        video_feature_path = os.path.join(video_folder, 'video_feature.pkl')
        if not os.path.exists(video_feature_path):
            video_features = []
            timestamps = [0.0] + timestamps
            s_t = time.time()
            for i in range(1, len(timestamps)):
                features = self.video_feature_extractor.get_features(
                    video_path=data['video_name'], start_time=timestamps[i-1], end_time=timestamps[i])
                video_features += [features]
            print("Video extraction time:", time.time() - s_t)
            torch.save(torch.stack(video_features), video_feature_path)
        else:
            # print("Read video feature", video_feature_path)
            video_features = torch.load(video_feature_path)
        # import ipdb;
        # ipdb.set_trace()

        # Not enough valid vecotors
        if len(vecs) < MIN_SHOTS_NUM:
            return [], [], []
        return vecs, ts, video_features

    def encode_gt(self, data, timestamp, video):
        """
        data: {
            "timestamp": "topic",
        }
        """

        gt = torch.zeros(len(timestamp), dtype=torch.long)

        # Nainpy of finding matches
        gt_ts_dict = {}
        for gt_ts in data.keys():
            gt_ts = float(gt_ts)
            # For each gt timestamp, find a matching segment (assuming there is)
            min_d = 100000  # no video longer than  this seconds
            min_idx = -1
            for idx, ts in enumerate(timestamp):
                if abs(gt_ts - ts) < min_d:
                    min_d = abs(gt_ts - ts)
                    min_idx = idx

            if min_idx == -1:
                print(f"issue with this {video}. some segmnet not assigned {gt_ts}")

            _, cur_min_d = gt_ts_dict.get(gt_ts, (0, 100000))

            if cur_min_d > min_d:
                gt_ts_dict[gt_ts] = (min_idx, min_d)

        for min_idx, min_d in gt_ts_dict.values():
            gt[min_idx] = 1
        # Start of the video always a start
        gt[0] = 1

        return gt

    def shuffle(self):
        data = list(zip(self.features, self.gts, self.timestamps, self.raw_gts, self.video_fearures))
        random.shuffle(data)
        self.features, self.gts, self.timestamps, self.raw_gts, self.video_fearures = zip(*data)

    def get_ts(self, idx: int) -> List[float]:
        return self.timestamps[idx]

    def get_raw_gt(self, idx: int) -> List[float]:
        return self.raw_gts[idx]

















