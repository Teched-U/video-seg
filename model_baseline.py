#!/user/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class SeqModel(nn.Module):
    def __init__(
        self,
        ntoken=None,
        nsrc=None,
        ninp=None,
        nhead=None,
        nhid=None,
        nlayers=None,
        dropout=0.5,
        seperate_feat=False,
        dim_dict=None,
    ):
        super(SeqModel, self).__init__()

        self.model_type = "Transformer"
        if seperate_feat:
            self.feature_extract = torch.nn.Sequential(
                SeqFeatureModel(dim_dict),
                torch.nn.Linear(dim_dict["output"], nsrc // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(nsrc // 2, ninp),
            )
        else:
            self.feature_extract = torch.nn.Sequential(
                torch.nn.Linear(nsrc, nsrc // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(nsrc // 2, ninp),
            )

        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp

        # [num_shot, ninput=300] -> [num_shot, ntoken]
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.feature_extract(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output


class SeqFeatureModel(nn.Module):
    def __init__(self, dim_dict):
        super(SeqFeatureModel, self).__init__()
        self.heads = {}
        for name, dims in dim_dict["layers"].items():
            self.heads[name] = nn.LSTM(
                input_size=dims[0],
                hidden_size=dims[1],
                bidirectional=True,
                batch_first=True,
            )

    def forward(self, src):
        feat_arr = []
        for key, input_seq in src.items():
            # Transform list of tensors to tensor:[seq_len, batch_size, input_size]
            if key == "asr":
                seq_lengths = torch.LongTensor(list(map(len, input_seq)))
                seq_tensor = torch.FloatTensor(
                    torch.zeros(len(input_seq), seq_lengths.max(), 300)
                )
                for idx, (seq, seqlen) in enumerate(zip(input_seq, seq_lengths)):
                    seq = torch.stack(seq)
                    seq_tensor[idx, :seqlen] = seq

                seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
                seq_tensor = seq_tensor[perm_idx]
                packed_input = pack_padded_sequence(
                    seq_tensor, seq_lengths.cpu().numpy(), batch_first=True
                )
                packed_output, (ht, ct) = self.heads[key](packed_input)
                h1 = ht[0]
                h2 = ht[1]
                # [batch size, dim_hid]
                feat_arr.append(torch.cat((h1, h2), dim=1))
            elif key == "video":
                input_seq = torch.stack(input_seq)
                batch, seq_len, input_size = input_seq.shape

                output_seq, (ht, ct) = self.heads[key](input_seq)

                # Get the last feature vec of both directions
                h1 = ht[0]
                h2 = ht[1]
                feat_arr.append(torch.cat((h1, h2), dim=1))
            else:
                # Alreayd a single feature vector, just stack them
                input_seq = torch.stack(input_seq)
                feat_arr.append(input_seq)

        # [batch_size, feature_dim]
        try:
            x = torch.cat(feat_arr, dim=1)
        except Exception:
            print("Issue with dimensions...")

        # [seq, 1, feature_dim]
        x = x.unsqueeze(1)

        return x
