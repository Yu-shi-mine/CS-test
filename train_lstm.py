"""
学習用
"""
 
import os, time
from datetime import datetime
from typing import OrderedDict
import warnings

import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F

 
from datahandler.lstm_dataset import LstmDataset
from model.lstm import LSTMSeq

warnings.simplefilter('ignore')

def train(net: nn.Module, dataloaders_dict: dict, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, num_epochs: int, log_dir: str):
    # Output folder
    dt = datetime.today().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, dt)
    os.makedirs(log_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # Iteration counter
    train_iteration = 1
    val_iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    step_duration = 0.0
    epoch_duration = 0.0
    disp = 'train'

    # Epoch loop
    # Record time first epoch start
    t_epoch_start = time.time()
    for epoch in range(num_epochs):

        print('--------------------------------------')
        print('Epoch{}/{}'.format(epoch+1, num_epochs))

        # Train and val loop
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                disp = 'train'
            else:
                net.eval()
                disp = 'valid'
            dataloader_list = dataloader_dict[phase]

            for dataloader in dataloader_list:
                # t_step_start = time.time()
                # minibatch loop
                for x, y in dataloader:
                    x = x.to(device)
                    y = y.to(device)

                    # Initialize optimizer
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):

                        output = net(x)

                        # Caluculate loss
                        loss = F.mse_loss(output, y)

                        # Backpropagation when phase == train
                        if phase == 'train':
                            loss.backward()

                            nn.utils.clip_grad_value_(
                                net.parameters(), clip_value=2.0
                            )

                            optimizer.step()

                            epoch_train_loss += loss.item()
                            train_iteration += 1
                        else:
                            epoch_val_loss =+ loss.item()
                            val_iteration += 1

                    
                    # Record time first epoch start
                    # t_step_finish = time.time()
                    # step_duration = t_step_finish - t_step_start
                    # t_step_start = time.time()

        scheduler.step()

        t_epoch_finish = time.time()
        epoch_duration = t_epoch_finish - t_epoch_start
        print('epoch: {} || loss: {:e} || val_loss: {:e}'.format(epoch+1, epoch_train_loss/train_iteration, epoch_val_loss/val_iteration))
        print('timer: {:.4f} sec.'.format(epoch_duration))
        t_epoch_start = time.time()
        train_iteration = 1
        val_iteration = 1

        # Save logs
        log_epoch = {
            'epoch': epoch+1,
            'loss': epoch_train_loss,
            'val_loss': epoch_val_loss
        }
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(log_dir, 'log.csv'))

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Save checkpoints every 10 epochs
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), os.path.join(log_dir, './weights_' + str(epoch+1) + '.pth'))


# Test
if __name__ == '__main__':
    # Settings
    dataset_dir = './datasets/06_20220703_213905_7feature_train30_val10_test1'
    feature_length = 7
    input_feature_length = 6
    output_feature_length = 7
    sequence_length = 90
    batch_size = 1
    min_max_scaler = MinMaxScaler(feature_length=feature_length)
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)

    #   Create DataLoader List
    phase = 'train'
    train_dataloader_list = GenDataLoaderList(
        dataset_dir=dataset_dir,
        phase=phase,
        feature_length=feature_length,
        input_feature_length=input_feature_length,
        output_feature_length=output_feature_length,
        sequence_length=sequence_length,
        batch_size=batch_size,
        transform=min_max_scaler
        )

    phase = 'val'
    val_dataloader_list = GenDataLoaderList(
        dataset_dir=dataset_dir,
        phase=phase,
        feature_length=feature_length,
        input_feature_length=input_feature_length,
        output_feature_length=output_feature_length,
        sequence_length=sequence_length,
        batch_size=batch_size,
        transform=min_max_scaler
        )

    dataloader_dict = {'train': train_dataloader_list, 'val': val_dataloader_list}

    # Define network
    net = LSTMSeq(input_feature_length=input_feature_length, output_feature_length=output_feature_length)

    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Define LRscheduler
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # train
    num_epochs = 100
    train(
        net=net,
        dataloaders_dict=dataloader_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        log_dir=log_dir
    )


