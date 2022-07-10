"""
学習用
"""

import os, time, glob
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F

 
from datahandler.lstm_dataset import LstmDataset
from model.lstm import LSTMSeq

warnings.simplefilter('ignore')

def train(net: nn.Module, dataloader_dict: dict, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, num_epochs: int, log_dir: str):
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
            else:
                net.eval()
            
            for x, y in dataloader_dict[phase]:
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
    DATASET_ROOT = './datasets/02_20220710_213723_sincos'
    
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)

    # Create Dataset
    train_path_list = glob.glob(os.path.join(DATASET_ROOT, 'train/*/joint/joint.csv'))
    val_path_list = glob.glob(os.path.join(DATASET_ROOT, 'val/*/joint/joint.csv'))

    window_size = 10
    runup_length  = 10
    inertia_length = 10

    train_datast = LstmDataset(
        csv_path_list=train_path_list,
        window_size=window_size,
        runup_length= runup_length,
        inertia_length=inertia_length
    )
    val_dataset = LstmDataset(
        csv_path_list=val_path_list,
        window_size=window_size,
        runup_length=runup_length,
        inertia_length=inertia_length
    )

    # Create DataLoader
    batch_size = 32
    train_dataloader = DataLoader(
        dataset=train_datast,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}

    # Define network
    input_size = 2
    lstm_input = 100
    hidden_size = 200
    output_size = 2
    net = LSTMSeq(
        input_size=input_size,
        lstm_input=lstm_input,
        hidden_size=hidden_size,
        output_size=output_size
    )

    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # Define LRscheduler
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # train
    num_epochs = 100
    train(
        net=net,
        dataloader_dict=dataloader_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        log_dir=log_dir
    )


