
import time

from dataset import VideoSegDataset 
from model_baseline import SeqModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math




def train_epoch(model, data_set, optimizer, criterion, epoch, scheduler, start_time)->None:
    model.train()
    total_loss = 0
    for i_sample, sample in enumerate(data_set):
        x, y = sample

        # Hack to mimic batch of 1
        x = x.unsqueeze(1)

        optimizer.zero_grad()
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
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i_sample, len(daa_set), scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    
DATA_FOLDER="/home/techedu/video-seg/data/easytopic"
RESULT_FOLDER="/home/techedu/video-seg/data/easytopic-gt/ground_truths"
NUM_EPOCH = 30

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = NUM_EPOCH

    ntoken = 2 # Binary 1, 0 for starting
    emsize = 300 # this is controled by how words are vectorized
    nhid = 200 # dimension of the feedforward network model in transfoer
    nlayers = 2 # number of encoderlayer 
    nhead = 2 # number of multi head
    dropout = 0.2 

    model = SeqModel(ntoken, emsize, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


    data_set = VideoSegDataset(
        data_folder=DATA_FOLDER,
        result_folder=RESULT_FOLDER,
        )

    for epoch in range(1, epochs + 1):
        # Manual shuffle (Use a dataloader later)
        data_set.shuffle()
        epoch_start_time = time.time()
        train_epoch(model, data_set, optimizer, criterion, epoch, scheduler, epoch_start_time)

        # val_loss = evaluate(model, val_data)
        # print('-' * 89)
        # print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #       'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                  val_loss, math.exp(val_loss)))
        # print('-' * 89)
    
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = model
    
        scheduler.step()


if __name__ == '__main__':
    train()
