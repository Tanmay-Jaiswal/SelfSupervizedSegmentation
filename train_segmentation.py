import torch
import os
import argparse
from torch_datasets import TumorSegDataset
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

    dataset_test = TumorSegDataset(config.datadir+"/test_data/", None)

    indices = torch.randperm(len(dataset_test)).tolist
    for i in range(0,1000,200):
        dataset_train_sup = torch.utils.data.Subset(dataset_test, indices[i:i + 200])
        remaining = [j for j in range(len(indices)) if j < i or j >= i + 200]
        dataset_test_k = torch.utils.data.Subset(dataset_test, remaining)
        # define training and validation data loaders
        data_loader_mini = torch.utils.data.DataLoader(
            dataset_train_sup, batch_size=8, shuffle=True, num_workers=2, )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test_k, batch_size=8, shuffle=False, num_workers=2, )

        train(config, data_loader_mini, data_loader_test)
