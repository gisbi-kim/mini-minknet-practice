import os 
import sys 
import time
import numpy as np 
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import MinkowskiEngine as ME

from models.model_utils import * 
from dataset.dataloader import *


def batch_to_device(data, device, verbose=False):
    coords, feats, labels = data
    coords = coords.to(device)
    feats = feats.to(device)
    labels = labels.to(device)

    if verbose:
        print(coords.shape)
        print(feats.shape)
        print(labels.shape)

    return coords, feats, labels


def epoch_train(model, train_dataloader, criterion, optimizer, scheduler, device):
    model.train()
    for n_iter, data in enumerate(train_dataloader):
        coords, feats, labels = batch_to_device(data, device, verbose=False)
        input = ME.SparseTensor(feats.float(), coords)
        output = model(input).F.squeeze() # SparseTensor to torch tensor 

        optimizer.zero_grad()
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        print("loss:", "{:5.5f}".format(loss.item()))


def epoch_test(epoch, model, train_dataloader, criterion, optimizer, scheduler, device):
    model.eval()
    for n_iter, data in enumerate(train_dataloader):
        coords, feats, labels = batch_to_device(data, device, verbose=False)
        input = ME.SparseTensor(feats.float(), coords)
        output = model(input).F.squeeze() # SparseTensor to torch tensor 

        optimizer.zero_grad()
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        print("loss:", "{:5.5f}".format(loss.item()))

        xyz = feats.cpu().numpy()
        rgb_gt = labels.cpu().numpy()
        rgb_est = output.detach().cpu().numpy()
        np.savetxt("pred/e" + str(epoch) + "_xyzrgb_est.txt", np.hstack([xyz,rgb_est]))
        np.savetxt("pred/e" + str(epoch) + "_xyzrgb_gt.txt", np.hstack([xyz,rgb_gt]))


def train():
    
    # device  
    device = torch.device('cuda')
    torch.cuda.empty_cache()

    # model 
    model = MinkUNet14(in_channels=3, out_channels=3, D=3).to(device)
    # print(model)

    # optim
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    scheduler_interval = 10

    # loss 
    criterion = nn.SmoothL1Loss()

    # a data loader must return a tuple of coords, features, and labels.
    train_dataloader = get_dataloader(batch_size=3, collate=True)

    eval_period = 25
    for epoch in tqdm(range(101)):
        if(epoch % eval_period == 0):
            epoch_test(epoch, model, train_dataloader, criterion, optimizer, scheduler, device)
        else:
            epoch_train(model, train_dataloader, criterion, optimizer, scheduler, device)



if __name__ == '__main__':
    train()