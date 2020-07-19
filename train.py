
import time

from dataset import VideoSegDataset 
from metric import evaluate_metrics
from model_baseline import *
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import random
import numpy as np
import os

import sys
# In order to reproduce the experiement
GRID_SEARCH= False
seed = 7793799
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
MODEL_PATH = 'checkpoint'
if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)

def train_epoch(model, data_set, optimizer, criterion, epoch, scheduler, start_time)->None:
    model.train()
    total_loss = 0
    for i_sample, sample in enumerate(data_set):
        # import ipdb; ipdb.set_trace()
        # print(i_sample, sample)
        """
            x (num_shot=200 ,D=300): [
                [1.0, 0.2, 1.0....], --> one shot
                [0.0, .0....]
                ....
                (num_shot = 200)
            ]

            y (num_shot=200) [
                1, 0, 1, 0....
            ]
        """
        x, y = sample

        # Hack to mimic batch of 1
        x = x.unsqueeze(1)

        optimizer.zero_grad()

        # output.shape = y.shape
        output = model(x)

        loss = criterion(output.view(-1, 2), y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        # log every 5 samples
        log_interval = 5
        if i_sample % log_interval == 0 and i_sample > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} samples | '
                  'lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i_sample, len(data_set), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(model :VideoSegClassificationModel, data: VideoSegDataset, criterion):
    model.eval()
    total_loss = 0.
    total_recall = 0.
    total_precision = 0.
    total_fscore = 0.
    with torch.no_grad():
        for i, sample in enumerate(data):
            x, y =  sample
            # Hack to mimic batch of 1
            x = x.unsqueeze(1)
            output = model(x)
            
            _, predicted = torch.max(output, 2)
            ts = data.get_ts(i)
            raw_gt = data.get_raw_gt(i)

            # Remove the batch dim
            predicted = predicted.squeeze()

            recall, precision, fscore = evaluate_metrics(predicted, ts, raw_gt)

            output_flat = output.view(-1, 2)
            total_loss += criterion(output_flat, y).item()
            total_recall += recall
            total_precision += precision
            total_fscore += fscore

    return total_loss/len(data), total_recall/len(data), total_precision/len(data), total_fscore/len(data)


DATA_ROOT = '/home/techedu'
DATA_FOLDER="/home/techedu/video-seg/data/easytopic"
# DATA_FOLDER = "/data/demo_videos_input"
RESULT_FOLDER="/home/techedu/video-seg/data/easytopic-gt/ground_truths"
NUM_EPOCH = 100

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = NUM_EPOCH

    ntoken = 2 # Binary 1, 0 for starting
    emsize = 300 # this is controled by how words are vectorized
    nhid = 200 # dimension of the feedforward network model in transfoer
    nlayers = 2 # number of encoderlayer
    nhead = 2 # number of multi head
    dropout = 0.2

    # model = SeqModel(ntoken, emsize, nhead, nhid, nlayers, dropout).to(device)
    model = VideoSegClassificationModel().to(device)
    # import ipdb


    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 5]))
    # lr = 0.01 # learning rate
    lr = 1.0  # learning rate
    print(model.parameters())

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5.0, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20.0, gamma=0.1)


    train_data = VideoSegDataset(
        data_folder=DATA_FOLDER,
        result_folder=RESULT_FOLDER,
        )

    val_data = train_data

    max_precision = max_f = 0.0
    max_pre_epoch = 1
    best_model = None
    for epoch in range(1, epochs + 1):
        # Manual shuffle (Use a dataloader later)
        train_data.shuffle()
        epoch_start_time = time.time()

        train_epoch(model, train_data, optimizer, criterion, epoch, scheduler, epoch_start_time)


        val_loss, recall, precision, fscore = evaluate(model, val_data, criterion)
        if precision > max_precision:
            max_precision = precision
            max_pre_epoch = epoch
            # Save the best model
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'best_model.pd'))

        if fscore > max_f:
            max_f = fscore
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'best_f_model.pd'))

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f} | recall: {:5.2f} | precision: {:5.5f} | fscore: {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss), recall, precision, fscore))
        print('-' * 89)
    
        scheduler.step()
        if epoch % 2 == 0:
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, f'model_epoch{epoch}.pd'))

    print('max_precision, fscore, epoch:', max_precision, max_f, max_pre_epoch)
    return max_precision

if __name__ == '__main__':
    if GRID_SEARCH:
        rand_ints = np.random.randint(low=0, high=9999999, size=100)
        best_acc = best_seed = 0
        for i in rand_ints:
            seed = i
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            res = train()
            if res > best_acc:
                best_acc = res
                best_seed = i
        print("Best seed and acc", best_seed, best_acc)
    else:
        train()