
import time
from model_baseline import *
import torch
import torch.nn as nn
import numpy as np
from model_baseline import *

import pickle
import os
from transformers import BertTokenizer, BertModel
import torch
import time
from video_feature_extractor import *

MIN_SHOTS_NUM = 4
class VideoSegPrediction():
    def __init__(self, seg_model_path=None):
        # Init bert
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # Video feature extractor
        self.video_feature_extractor = VideoFeatureExtractor()

        # Get features and labels
        self.features = []

        # Timestamps for each video
        self.timestamps = []

        # video features
        self.video_fearures = []

        # Name
        self.NAME = None

        # Seg model init
        self.model = VideoSegClassificationModel()
        if seg_model_path is None:
            self.model.load_state_dict(torch.load('./checkpoint/best_model.pd'))
        else:
            self.model.load_state_dict(torch.load(seg_model_path))
        self.model.eval()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Return:
            featurs tensor: tensor[num_shot, D=300]
            gt tensor: tensor[num_shot, (0,1)]
        """
        feature_tensor = torch.stack(self.features[idx])
        video_tensor = self.video_fearures[idx]
        return torch.cat([feature_tensor, video_tensor], dim=1), self.gts[idx]


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

            # inputs = self.bert_tokenizer.tokenize(input_str, return_tensors="pt")
            # if len(inputs) > 510: inputs = inputs[:510]
            # return self.bert_model(**inputs)[0][0][0]
            return last_hidden_states[0][0]

    def encode_features(self, data):
        sentences = [shot["transcript"] for shot in data['features']]
        timestamps = [shot["timestamp"] for shot in data['features']]
        self.timestamps = timestamps

        video_folder = os.path.dirname(data['video_name'])
        bert_feature_path = os.path.join(video_folder, 'bert_feature.pkl')
        if not os.path.exists(bert_feature_path):
            print("Run Bert...")
            s_t = time.time()
            result = [self.run_bert(sentence) for sentence in sentences]
            print("Time cost:", time.time() - s_t)
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
        video_feature_path = os.path.join(video_folder, 'video_feature.pkl')
        if not os.path.exists(video_feature_path):
            video_features = []
            timestamps = [0.0] + timestamps
            s_t = time.time()
            print("Run video extractor...")
            for i in range(1, len(timestamps)):
                features = self.video_feature_extractor.get_features(
                    video_path=data['video_name'], start_time=timestamps[i-1], end_time=timestamps[i])
                video_features += [features]
            print("Video extraction time:", time.time() - s_t)
            torch.save(torch.stack(video_features), video_feature_path)
        else:
            # print("Read video feature", video_feature_path)
            video_features = torch.load(video_feature_path)


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

    def test_one_video(self, video_path):
        # self.NAME = '04_special-applications-face-recognition-neural-style-transfer'
        self.NAME = os.path.basename(video_path).split('.')[0]
        self.ROOT = os.path.dirname(video_path)
        self.JSON_PATH = os.path.join(self.ROOT, self.NAME+".json")

        input = None
        data = {}
        with open(self.JSON_PATH, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception:
                print(f"error loading {self.JSON_PATH}")
            asr_feature, timestamp, video_feature = self.encode_features(data)

        if asr_feature:
            # import ipdb; ipdb.set_trace()
            if type(video_feature) == list:
                video_feature = torch.stack(video_feature)
            input = torch.cat([torch.stack(asr_feature), video_feature], dim=1)
            res = self.model(input)
            res = torch.argmax(res, axis=1).numpy()

            # import ipdb; ipdb.set_trace()
            target_idx = np.argwhere(res == 1)
            print(target_idx)
            target_idx.reshape(-1)

            print(np.array(self.timestamps)[res == 1])

            # save result
            with open(self.JSON_PATH, "wb") as f:
                data['Seg'] = res.tolist()
                pickle.dump(data, f)
            return res
        else:
            print("Bad video")

    def export_new_video(self):
        return


if __name__ == "__main__":
    video_model = VideoSegPrediction('./checkpoint/model_epoch68.pd')
    res = video_model.test_one_video('/data/easytopic-data/2PhaT6AbH3Q/2PhaT6AbH3Q.mp4')
    # res = video_model.test_one_video('/data/demo_videos/operating_system_demo_video1.mp4')
    print(res)
    print("Shape:", np.shape(res))






