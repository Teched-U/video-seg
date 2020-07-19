import time

from dataset import VideoSegDataset
from metric import evaluate_metrics
from model_baseline import SeqModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import click


def train_epoch(
    model, data_set, optimizer, criterion, epoch, scheduler, start_time, seperate_feat
) -> None:
    model.train()
    total_loss = 0
    for i_sample, sample in enumerate(data_set):
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
        if not seperate_feat:
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
            print(
                "| epoch {:3d} | {:5d}/{:5d} samples | "
                "lr {:5.5f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    i_sample,
                    len(data_set),
                    scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


def evaluate(model: SeqModel, data: VideoSegDataset, criterion):
    model.eval()
    total_loss = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_fscore = 0.0
    with torch.no_grad():
        for i, sample in enumerate(data):
            x, y = sample

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

    return (
        total_loss / len(data),
        total_recall / len(data),
        total_precision / len(data),
        total_fscore / len(data),
    )


DATA_FOLDER = "/home/techedu/video-seg/data/easytopic"
RESULT_FOLDER = "/home/techedu/video-seg/data/easytopic-gt/ground_truths"
NUM_EPOCH = 80


def train(seperate_feat, load, save, num_frame) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = NUM_EPOCH
    ntoken = 2  # Binary 1, 0 for starting
    emsize = (
        VideoSegDataset.FEATURE_SIZE
    )  # this is controled by how words are vectorized
    nhid = 128  # dimension of the feedforward network model in transfoer
    nsrc = 256  # reduced dimension to extract features
    nlayers = 1  # number of encoderlayer
    nhead = 1  # number of multi head
    dropout = 0.2

    dim_dict = {
        "layers": {"video": [1280, 256], "asr": [300, 64],},
        "output": 256 * 2 + 64 * 2 + 4,
    }

    # FIXME(oyzh): check the networks
    model = SeqModel(
        ntoken,
        emsize,
        nsrc,
        nhead,
        nhid,
        nlayers,
        dropout,
        seperate_feat,
        dim_dict=dim_dict,
    ).to(device)

    # FIXME(oyzh): check if these training parameters make sense
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 8]))
    lr = 0.1  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # FIXME(oyzh): take a look at how the dataset is organized
    train_data = VideoSegDataset(
        data_folder=DATA_FOLDER,
        result_folder=RESULT_FOLDER,
        seperate=seperate_feat,
        load_dir=load,
        save_dir=save,
        num_frame=num_frame,
    )

    val_data = train_data

    for epoch in range(1, epochs + 1):
        # Manual shuffle (Use a dataloader later)
        # train_data.shuffle()
        epoch_start_time = time.time()
        train_epoch(
            model,
            train_data,
            optimizer,
            criterion,
            epoch,
            scheduler,
            epoch_start_time,
            seperate_feat,
        )

        val_loss, recall, precision, fscore = evaluate(model, val_data, criterion)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f} | recall: {:5.2f} | precision: {:5.2f} | fscore: {:5.2f}".format(
                epoch,
                (time.time() - epoch_start_time),
                val_loss,
                math.exp(val_loss),
                recall,
                precision,
                fscore,
            )
        )
        print("-" * 89)

        scheduler.step()


@click.command()
@click.option(
    "-s",
    "--seperate-feat",
    "seperate_feat",
    is_flag=True,
    help="Seperating features (not using mean)",
)
@click.option("--load-feat", "load", metavar="DIR", help="Loading featurs from DIR")
@click.option("--save-feat", "save", metavar="DIR", help="Save features to DIR")
@click.option(
    "--num-frame",
    "num_frame",
    type=int,
    default=5,
    help="Number of frames to extract from a shot in video feature generation",
)
def main(seperate_feat, load, save, num_frame):
    train(seperate_feat, load, save, num_frame)


if __name__ == "__main__":
    main()
