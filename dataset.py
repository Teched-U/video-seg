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

# Max length of a sentecnce in a shot
MAX_LENGTH=128
MIN_SHOTS_NUM = 16

GOOGLE_MODEL_PATH = '/media/word2vec/GoogleNews-vectors-negative300.bin'
STOPWORD_PATH = 'data/stopwords_en.txt'


class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
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
            return False ,np.zeros((300,))

        vector = np.mean(word_vecs, axis=0)

        vector = torch.from_numpy(vector)

        return True, vector

def init_word2vec(model_path:str, stopwords_file:str) -> Tuple[DocSim, List[str]] :
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

        # Init Word2Vec model

        self.docsim_model, self.stopwords = init_word2vec(GOOGLE_MODEL_PATH, STOPWORD_PATH)

        # Get features and labels 
        self.features = [] 
        self.gts = []
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
                feature, timestamp = self.encode_features(data)
                if feature:
                    self.features.append(feature)
                else:
                    # Invalid
                    continue

            # Get label
            video_name = os.path.basename(data_file)
            gt_file = os.path.join(result_folder, f"gt_{video_name}")
            with open(gt_file, "r") as f:
                gt_data= json.load(f)
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
        return torch.stack(self.features[idx]), self.gts[idx]


    def encode_features(self, data):
        sentences = [shot["transcript"] for shot in data['features']]
        timestamps = [shot["timestamp"] for shot in data['features']]

        # 300 dim : feature vector
        result = [self.docsim_model.vectorize(sentence) for sentence in sentences]
        vecs = []
        ts = []

        for res, t in zip(result, timestamps):
            valid, vec = res
            if valid:
                vecs.append(vec)
                ts.append(t)

        # TODO(OY): encode video features

        # Not enough valid vecotors
        if len(vecs) < MIN_SHOTS_NUM:
            return [], []
        return vecs, ts 
    

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
            min_d = 100000 # no video longer than  this seconds 
            min_idx = -1
            for idx, ts in enumerate(timestamp):
                if abs(gt_ts - ts) < min_d:
                    min_d = abs(gt_ts -ts)
                    min_idx = idx

            if min_idx == -1:
                print(f"issue with this {video}. some segmnet not assigned {gt_ts}") 

            _, cur_min_d = gt_ts_dict.get(gt_ts, (0, 100000))

            if cur_min_d > min_d:
                gt_ts_dict[gt_ts] = (min_idx, min_d)
            

        for min_idx, min_d in gt_ts_dict.values():
            gt[min_idx] = 1
        # Start of the video always a start
        gt[0]  = 1

        return gt

    def shuffle(self):
        data = list(zip(self.features, self.gts))
        random.shuffle(data)
        self.features, self.gts = zip(*data)



                





        


        





