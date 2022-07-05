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
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datahandler.lstm_dataset import JointDataset
from model.lstm import LSTMSeq

warnings.simplefilter('ignore')

def train(net: nn.Module, dataloaders_dict: dict, optimizer:torch.optim, num_epochs: int, log_dir: str):
    # Output folder
    dt = datetime.today().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir, dt)
    os.makedirs(log_dir, exist_ok=True)

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # Iteration counter
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    step_duration = 0.0
    epoch_duration = 0.0
    disp = 'train'

    # Epoch loop
    for epoch in range(num_epochs):
        # Timer
        t_epoch_start = time.time()

        # Train and val loop
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                disp = 'train'
            else:
                net.eval()
                disp = 'valid'

            # minibatch loop
            with tqdm(dataloaders_dict[phase]) as pbar:
                pbar.set_description(f'[Epoch {epoch+1}/{num_epochs} : phase={disp}]')

                # timer
                t_step_start = time.time()
                for x, y in pbar:
                    

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
                            iteration += 1
                        else:
                            epoch_val_loss += loss.item()
                    
                    # timer
                    t_step_finish = time.time()
                    step_duration = t_step_finish - t_step_start
                    t_step_start = time.time()
        
        # Loss and accuracy at every epoch
        t_epoch_finish = time.time()
        print('epoch: {} || loss: {:.4f} || val_loss: {:.4f}'.format(epoch+1, epoch_train_loss, epoch_val_loss))
        t_epoch_start = time.time()

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
    dataset_dir = './datasets/20220702_084913'
    feature_length = 1
    sequence_length = 2
    batch_size = 1
    transform = None
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)

    #  Generate datasets for training
    phase = 'train'
    train_dataset = JointDataset(
        feature_length=feature_length,
        dataset_dir=dataset_dir, 
        phase=phase, 
        sequence_length=sequence_length
        )
    
    # Generate datasets for validation
    phase = 'val'
    val_dataset = JointDataset(
        feature_length=feature_length,
        dataset_dir=dataset_dir, 
        phase=phase, 
        sequence_length=sequence_length
        )

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    # Define network
    net = LSTMSeq()

    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # train
    num_epochs = 100
    train(
        net=net,
        dataloaders_dict=dataloaders_dict,
        optimizer=optimizer,
        num_epochs=num_epochs,
        log_dir=log_dir
    )


