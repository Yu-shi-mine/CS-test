"""
Train LSTM
"""

from typing import Tuple
import warnings
from datetime import datetime
import os, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from dataset import WaveDatasetOne2Many, WaveDatasetOne2One
from model.model import LSTMseq, LSTMStateful, SimpleLSTM


warnings

@hydra.main(config_name='config', config_path='config')
def main(cfg: DictConfig):
    # Output folder
    dt = datetime.today().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(cfg.train.log_root, dt)
    os.makedirs(log_dir, exist_ok=True)

    # Start mlflow tracking
    with mlflow.start_run(run_name=cfg.run_name):
        # Track mlflow params
        log_param('sequence_num', cfg.dataset.sequence_num)
        log_param('data_window', cfg.dataset.data_window)
        log_param('label_window', cfg.dataset.label_window)
        log_param('total_wave_num', cfg.dataset.total_wave_num)
        log_param('batch_size', cfg.dataloader.batch_size)
        log_param('num_epochs', cfg.train.num_epochs)
        log_param('input_size', cfg.model.input_size)
        log_param('hidden_size', cfg.model.hidden_size)
        log_param('output_size', cfg.model.output_size)

        # Define model
        if cfg.model.name == 'One2One':
            model = LSTMStateful(
                input_size=cfg.model.input_size,
                hidden_size=cfg.model.hidden_size,
                output_size=cfg.model.output_size
            )
        elif cfg.model.name == 'One2Many':
            model = LSTMseq(
                input_size=cfg.model.input_size,
                hidden_size=cfg.model.hidden_size,
                output_size=cfg.model.output_size
            )
        else:
            model = SimpleLSTM(
                input_size=cfg.model.input_size,
                hidden_size=cfg.model.hidden_size,
                output_size=cfg.model.output_size
            )

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Define LR scheduler
        scheduler = ExponentialLR(optimizer, gamma=0.95)

        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        torch.backends.cudnn.benchmark = True

        # Create Dataset
        dataloaders_dict = gen_dataloaders(cfg)
        
        # Iteration counter
        train_iteration = 1
        val_iteration = 1
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        logs = []
        epoch_duration = 0.0

        # Epoch loop
        # Record time first epoch start
        t_epoch_start = time.time()
        for epoch in range(cfg.train.num_epochs):
            print('--------------------------------------')

            # Train and val loop
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                # Mini batch Loop
                for i, (x, y) in enumerate(tqdm(dataloaders_dict[phase], desc=f'{epoch+1}/{cfg.train.num_epochs}')):
                    # Initialize optimizer
                    optimizer.zero_grad()

                    # Initialize hidden and cell state
                    if i % cfg.dataset.label_window == 0:
                        hn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
                        cn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])

                    # Send to device
                    x: torch.Tensor = x.to(device)
                    y: torch.Tensor = y.to(device)
                    hn: torch.Tensor = hn.to(device)
                    cn: torch.Tensor = cn.to(device)

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):

                        # outputs, hn, cn = model(x, hn, cn)
                        outputs= model(x)

                        # Caluculate loss
                        loss = F.mse_loss(outputs, y)

                        # Backpropagation when phase == train
                        if phase == 'train':
                            loss.backward()

                            nn.utils.clip_grad_value_(
                                model.parameters(), clip_value=2.0
                            )

                            # Step optimizer
                            optimizer.step()

                            epoch_train_loss += loss.item()
                            train_iteration += 1
                        else:
                            epoch_val_loss += loss.item()
                            val_iteration += 1

            # step LR scheduler
            scheduler.step()

            # Caluculate average loss
            train_loss = epoch_train_loss/train_iteration
            val_loss = epoch_val_loss/val_iteration

            # Save mlflow metrics
            log_metric('train_loss', train_loss)
            log_metric('val_loss', val_loss)

            # Display
            t_epoch_finish = time.time()
            epoch_duration = t_epoch_finish - t_epoch_start
            print('epoch: {} || loss: {:e} || val_loss: {:e}'.format(epoch+1, train_loss, val_loss))
            print('timer: {:.4f} sec.'.format(epoch_duration))
            t_epoch_start = time.time()
            train_iteration = 1
            val_iteration = 1

            # Save logs
            log_epoch = {
                'epoch': epoch+1,
                'loss': train_loss,
                'val_loss': val_loss
            }
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv(os.path.join(log_dir, 'log.csv'), index=0)

            epoch_train_loss = 0.0
            epoch_val_loss = 0.0

            

            # Save checkpoints every 10 epochs
            if ((epoch+1) % 10 == 0):
                torch.save(model.state_dict(), os.path.join(log_dir, './weights_' + str(epoch+1) + '.pth'))

    mlflow.end_run()


def gen_dataloaders(cfg: DictConfig) -> dict[str: DataLoader, str: DataLoader]:
    x = np.arange(0, 2*np.pi, 2*np.pi/cfg.dataset.sequence_num, dtype=np.float32)
    train_dataset = WaveDatasetOne2One(
        x=x,
        add_noise=cfg.dataset.add_noise,
        data_winow=cfg.dataset.data_window,
        label_window=cfg.dataset.label_window,
        time_shift=cfg.dataset.time_shift,
        total_wave_nums=cfg.dataset.total_wave_num
    )
    val_dataset = WaveDatasetOne2One(
        x=x,
        add_noise=cfg.dataset.add_noise,
        data_winow=cfg.dataset.data_window,
        label_window=cfg.dataset.label_window,
        time_shift=cfg.dataset.time_shift,
        total_wave_nums=cfg.dataset.total_wave_num
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_worklers
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_worklers
    )
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
    return dataloaders_dict


# Run
if __name__ =='__main__':
    main()