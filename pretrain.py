import torch
import os
import argparse
from torch_datasets import TumorRecDataset
from train_utils import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--datadir', type=str, default="data/")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_weight_path', type=str, default="snapshots/pretrain-{epoch}.pt")

    config = parser.parse_args()
    config.pretrain = True

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    dataset_train = TumorRecDataset(config.datadir+"/train_data/", None)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=8, shuffle=True, num_workers=2, )

    train(config, data_loader_train)
