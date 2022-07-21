"""
Test trained LSTM model
"""

from typing import Tuple
from datetime import datetime
import os

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


@hydra.main(config_name='config', config_path='config')
def main(cfg: DictConfig):
    # Output folder
    dt = datetime.today().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(cfg.infer.weight_path, 'result')
    os.makedirs(result_dir, exist_ok=True)

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

    # Load trained weights
    weights = torch.load(os.path.join(cfg.infer.weight_path, cfg.infer.weight_name), map_location={'cuda:0':'cpu'})
    model.load_state_dict(weights)

    # Create test DataLoader
    dataloader = gen_dataloader(cfg)

    # Infer
    if cfg.model.name == 'One2One':
        if cfg.infer.recursive:
            result = recursive_infer_one2one(cfg, model, dataloader)
        else:
            result = infer_one2one(cfg, model, dataloader)
    elif cfg.model.name == 'One2Many':
        if cfg.infer.recursive:
            result = recursive_infer_one2many(cfg, model, dataloader)
        else:
            result = infer_one2many(cfg, model, dataloader)
    else:
        if cfg.infer.recursive:
            result = recursive_infer_one2one(cfg, model, dataloader)
        else:
            result = infer_one2one(cfg, model, dataloader)
    
    # Save results
    df = pd.DataFrame(result, index=None, columns=['pred', 'truth'])
    df.to_csv(os.path.join(result_dir, f'result_{dt}.csv'))


def infer_one2one(cfg, model, dataloader) -> np.ndarray:
    with torch.no_grad():
        isFirst = True
        for x, y in dataloader:
            # Initialize hidden and cell state for the first step
            if isFirst:
                hn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
                cn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
            
            # infer
            outputs, hn, cn = model(x, hn, cn)

            # Save results
            pred = outputs[:, 0, :].numpy()
            truth = y.numpy()
            returns = np.concatenate([pred, truth], axis=1)

            if isFirst:
                result = returns
                isFirst = False
            else:
                result = np.concatenate([result, returns], axis=0)
    return result


def recursive_infer_one2one(cfg, model, dataloader) -> np.ndarray:
    with torch.no_grad():
        isFirst = True
        for x, y in dataloader:
            # Initialize hidden and cell state for the first step
            if isFirst:
                _input = x
                hn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
                cn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
            
            # infer
            # outputs, hn, cn = model(_input, hn, cn)
            outputs = model(_input)

            # Save results
            pred = outputs[:, 0, :].numpy()
            truth = y.numpy()
            returns = np.concatenate([pred, truth], axis=1)
            if isFirst:
                result = returns
                isFirst = False
            else:
                result = np.concatenate([result, returns], axis=0)

            # Prepare for next step
            previous_input = _input[:, 1:, :]
            next_head = outputs[:, :1, :]
            _input = torch.cat([previous_input, next_head], axis=1)
    return result


def infer_one2many(cfg, model, dataloader) -> np.ndarray:
    with torch.no_grad():
        isFirst = True
        for x, y in dataloader:
            # Initialize hidden and cell state for the first step
            if isFirst:
                hn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
                cn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
            
            # infer
            outputs, hn, cn = model(x, hn, cn)

            # Save results
            pred = outputs[:, 0, :].numpy()
            truth = y[:, 0, :].numpy()
            returns = np.concatenate([pred, truth], axis=1)

            if isFirst:
                result = returns
                isFirst = False
            else:
                result = np.concatenate([result, returns], axis=0)
    return result


def recursive_infer_one2many(cfg, model, dataloader) -> np.ndarray:
    with torch.no_grad():
        isFirst = True
        for x, y in dataloader:
            # Initialize hidden and cell state for the first step
            if isFirst:
                _input = x
                hn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
                cn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
            
            # infer
            outputs, hn, cn = model(_input, hn, cn)

            # Save results
            pred = outputs[:, 0, :].numpy()
            truth = y[:, 0, :].numpy()
            returns = np.concatenate([pred, truth], axis=1)
            if isFirst:
                result = returns
                isFirst = False
            else:
                result = np.concatenate([result, returns], axis=0)

            # Prepare for next step
            previous_input = _input[:, 1:, :]
            next_head = outputs[:, :1, :]
            _input = torch.cat([previous_input, next_head], axis=1)
    return result


def gen_dataloader(cfg: DictConfig) -> DataLoader:
    x = np.arange(0, 2*np.pi, 2*np.pi/cfg.dataset.sequence_num, dtype=np.float32)
    test_dataset = WaveDatasetOne2One(
        x=x,
        add_noise=cfg.dataset.add_noise,
        data_winow=cfg.dataset.data_window,
        label_window=cfg.dataset.label_window,
        time_shift=cfg.dataset.time_shift,
        total_wave_nums=cfg.dataset.total_wave_num
    )

    # Create DataLoader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_worklers
    )
    return test_dataloader


# Run
if __name__ =='__main__':
    main()