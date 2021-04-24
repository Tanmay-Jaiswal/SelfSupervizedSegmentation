import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from model import UNet
import matplotlib.pyplot as plt


def calc_accuracy(output, mask):
    output = torch.argmax(output, dim=1)
    return torch.mean((output == mask).float())


def train_one_epoch(model, optimizer, loss_func, data_loader, device, epoch, print_freq=10, acc_func=None):
    model.train()
    epoch_loss = []
    accuracy = []
    start_time = time.time()
    for i, (images, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        images = images.cuda()
        targets = targets.cuda()
        outputs = model(images)

        if loss_func is CombinedLoss:
            loss = loss_func(outputs, targets, images)
        else:
            loss = loss_func(outputs, targets)
        epoch_loss.append(loss.data.cpu())
        if acc_func:
            acc = acc_func(outputs, targets)
            accuracy.append(acc.data.cpu())
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(
                f"Epoch =  {epoch} \t Iter: {i} \t training_loss = {loss} \t time from epoch={time.time() - start_time}")

    epoch_loss = np.array(epoch_loss).mean()
    if acc_func:
        accuracy = np.array(accuracy).mean()
        return epoch_loss, accuracy

    return epoch_loss


@torch.no_grad()
def evaluate(model, data_loader, loss_func, device, print_freq=10, iters=200, acc_func=None, pr=True):
    model.eval()
    accuracy = []
    epoch_loss = []
    tp = 0
    fp = 0
    tot_p = 0
    # model_time = time.time()
    for i, (images, targets) in enumerate(data_loader):
        if i == iters:
            break
        images = images.cuda()
        targets = targets.cuda()
        outputs = model(images)
        # outputs = F.relu(outputs)
        # outputs = torch.sigmoid(outputs)

        loss = loss_func(outputs, targets)
        epoch_loss.append(loss.data.cpu())
        if acc_func:
            acc = acc_func(outputs, targets)
            accuracy.append(acc.data.cpu())
        if pr:
            tp += torch.sum(torch.logical_and(torch.argmax(outputs, dim=1), targets))
            fp += torch.sum(torch.logical_and(torch.argmax(outputs, dim=1), torch.logical_not(targets)))
            tot_p += torch.sum(targets)

        if i % print_freq == 0:
            print(f"Iter: {i} \t val_loss = {loss}")
    epoch_loss = np.array(epoch_loss).mean()
    if pr:
        recall = tp / tot_p
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        print("Recall = ", recall)
        print("Precision = ", precision)
        print("F1 score = ", f1)
    if acc_func:
        accuracy = np.array(accuracy).mean()
        return epoch_loss, accuracy

    return epoch_loss


class CombinedLoss(nn.Module):
    def __init__(self, weights):
        super(CombinedLoss).__init__()
        self.CELoss = nn.CrossEntropyLoss(weights)

    def forward(self, outputs, mask, input_image):
        loss = F.mse_loss(outputs[:, 2:], input_image)
        loss += self.CELoss(outputs[:, :2], mask)
        return loss


def train(config, dataloader_train, dataloader_test=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if config.pretrain:
        # 3 classes one for each color channel
        num_classes = 3
        model = UNet(n_channels=3, n_classes=num_classes)
        model.pretrain(True)
        loss_func = F.mse_loss
    else:
        # 3 classes one for each color channel and 2 for segmentation
        num_classes = 5
        model = UNet(n_channels=3, n_classes=num_classes)
        model.pretrain(False)
        # 0.3 and 0.7 were calculated by checking the ratio of pixels that give
        loss_func = CombinedLoss(weight=torch.tensor([0.3, 0.7], device="cuda"))

    # move model to the right device
    model.to(device)

    if config.load_pretrain:
        try:
            pretrained_dict = torch.load(config.pretrain_weight_path, map_location=device)

            # 1. filter out unnecessary keys
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "outc" not in k}

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            model.load_state_dict(model_dict)
        except Exception as e:
            print("Could not load weights")
            raise e

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(config.num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, optimizer, loss_func, dataloader_train, device, epoch,
                                                     print_freq=config.print_freq)
        print(f"Epoch =  {epoch} \t Train loss = {train_loss} \t Train Accuracy = {train_accuracy}")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # update the learning rate
        lr_scheduler.step()
        if config.pretrain:
            model_name = config.snapshots_folder + f"/pretrain-{epoch}.pt"
            torch.save(model.state_dict(), model_name)
            continue
        else:
            model_name = config.snapshots_folder + f"/model-{epoch}.pt"
            torch.save(model.state_dict(), model_name)

        val_loss, val_accuracy = evaluate(model, dataloader_test, loss_func, device=device, print_freq=20,
                                          acc_func=calc_accuracy, iters=20)
        print(f"Epoch =  {epoch} \t Validation loss = {val_loss} \t Validation Accuracy = {val_accuracy}")
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    plt.plot(train_accuracies, label="train accuracy")
    plt.plot(val_accuracies, label="val accuracy")
    plt.legend()
    plt.savefig(config.snapshots_folder + "/accuracy_plot.jpg")

    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.legend()
    plt.savefig(config.snapshots_folder + "/loss_plot.jpg")
