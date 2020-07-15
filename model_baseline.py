#!/user/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class SeqModel(nn.Module):
    def __init__(self, ntoken=None, ninp=None, nhead=None, nhid=None, nlayers=None, dropout=0.5):
        super(SeqModel, self).__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        # [num_shot, ninput] -> [num_shot, ntoken]
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output

# Simple merge feature and only one FC
class VideoSegClassificationModel(nn.Module):
    def __init__(self, in_dim=768 + 1280, out_dim=2):
        super(VideoSegClassificationModel, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x

## Use FC to merge features
# class VideoSegClassificationModel(nn.Module):
#     def __init__(self, in_dim=768+1280, out_dim=2):
#         super(VideoSegClassificationModel, self).__init__()
#         self.text_dim = 768
#         self.video_dim = 1280
#         hidden_layer_dim = 256
#
#         self.fc_audio = nn.Linear(self.text_dim, hidden_layer_dim)
#         self.fc_video = nn.Linear(self.video_dim, hidden_layer_dim)
#         self.last_fc = nn.Linear(hidden_layer_dim*2, out_dim)
#
#         self.fc1 = nn.Linear(in_dim, out_dim)
#
#     def forward(self, x):
#         audio_x = self.fc_audio(x[:, :, :self.text_dim])
#         vidio_x = self.fc_video(x[:, :, self.text_dim:])
#         # vidio_x = torch.relu(vidio_x)
#         x = torch.cat([audio_x, vidio_x], dim=-1)
#         x = self.last_fc(x)
#         return x

# class VideoSegClassificationModel(nn.Module):
#     def __init__(self, in_dim=768, out_dim=2):
#         super(VideoSegClassificationModel, self).__init__()
#
#
#         # self.fc1 = nn.Linear(in_dim, out_dim)
#         self.fc1 = nn.Linear(in_dim, 256)
#         # self.drop
#         self.fc2 = nn.Linear(256, 512)
#         self.fc3 = nn.Linear(512, out_dim)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = torch.relu(x)
#         x = self.fc2(x)
#         x = torch.relu(x)
#         x = self.fc3(x)
#         return x

