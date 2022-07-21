"""
Dataset Generators
"""

from typing import Tuple

import numpy as np
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset, DataLoader


class WaveDatasetOne2Many(Dataset):
    def __init__(self, x: np.ndarray, add_noise: bool, data_winow: int, label_window: int, time_shift: int, total_wave_nums: int) -> None:
        self.data, self.label = self.generate_data(x, add_noise, data_winow, label_window, time_shift, total_wave_nums)
    
    def generate_data(self, x: np.ndarray, add_noise: bool, data_winow: int, label_window: int, time_shift: int, total_wave_nums: int) -> Tuple[np.ndarray, np.ndarray]:
        isFirst = True
        for i in range(total_wave_nums):
            sinx = self.random_sin(x, add_noise)
            sub_data, sub_label = self.seq2window(sinx, data_winow, label_window, time_shift)
            if isFirst:
                data = sub_data
                label = sub_label
                isFirst = False
            else:
                data = np.concatenate([data, sub_data], axis=0)
                label = np.concatenate([label, sub_label], axis=0)
        return data, label

    def random_sin(self, x: np.ndarray, add_noise: bool) -> np.ndarray:
        sinx : np.ndarray = np.sin(x)
        if add_noise:
            noise = np.random.normal(loc=0., scale=0.05, size=x.shape)
            sinx = sinx + noise
        if sinx.ndim == 1:
            sinx = sinx[:, np.newaxis]
        return sinx

    def seq2window(self, sequence: np.ndarray, data_window: int, label_window: int,time_shift: int) -> Tuple[np.ndarray, np.ndarray]:
        isFirst = True
        for i in range((sequence.shape[0]-label_window) // time_shift):
            window_data = sequence[i:i+data_window, :]
            window_label = sequence[i+1:i+label_window+1, :]
            window_data = window_data[np.newaxis, :, :]
            window_label = window_label[np.newaxis, :, :]
            if isFirst:
                sub_data = window_data
                sub_label = window_label
                isFirst = False
            else:
                sub_data = np.concatenate([sub_data, window_data], axis=0)
                sub_label = np.concatenate([sub_label, window_label], axis=0)
        return sub_data, sub_label
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.label[index], dtype=torch.float32)
        return x, y

class WaveDatasetOne2One(Dataset):
    def __init__(self, x: np.ndarray, add_noise: bool, data_winow: int, label_window: int, time_shift: int, total_wave_nums: int) -> None:
        self.data, self.label = self.generate_data(x, add_noise, data_winow, label_window, time_shift, total_wave_nums)
    
    def generate_data(self, x: np.ndarray, add_noise: bool, data_winow: int, label_window: int, time_shift: int, total_wave_nums: int) -> Tuple[np.ndarray, np.ndarray]:
        isFirst = True
        for i in range(total_wave_nums):
            sinx = self.random_sin(x, add_noise)
            sub_data, sub_label = self.seq2window(sinx, data_winow, label_window, time_shift)
            if isFirst:
                data = sub_data
                label = sub_label
                isFirst = False
            else:
                data = np.concatenate([data, sub_data], axis=0)
                label = np.concatenate([label, sub_label], axis=0)
        return data, label

    def random_sin(self, x: np.ndarray, add_noise: bool) -> np.ndarray:
        sinx : np.ndarray = np.sin(x)
        if add_noise:
            noise = np.random.normal(loc=0., scale=0.05, size=x.shape)
            sinx = sinx + noise
        if sinx.ndim == 1:
            sinx = sinx[:, np.newaxis]
        return sinx

    def seq2window(self, sequence: np.ndarray, data_window: int, label_window: int,time_shift: int) -> Tuple[np.ndarray, np.ndarray]:
        isFirst = True
        for i in range((sequence.shape[0]-data_window) // time_shift):
            window_data = sequence[i:i+data_window, :]
            window_label = sequence[i+data_window, :]
            window_data = window_data[np.newaxis, :, :]
            window_label = window_label[np.newaxis, :]
            if isFirst:
                sub_data = window_data
                sub_label = window_label
                isFirst = False
            else:
                sub_data = np.concatenate([sub_data, window_data], axis=0)
                sub_label = np.concatenate([sub_label, window_label], axis=0)
        return sub_data, sub_label
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.label[index], dtype=torch.float32)
        return x, y


@hydra.main(config_name='config', config_path='config')
def main(cfg: DictConfig):
    # Create Dataset
    x = np.arange(0, 2*np.pi, 2*np.pi/cfg.dataset.sequence_num, dtype=np.float32)
    dataset = WaveDatasetOne2One(
        x=x,
        add_noise=cfg.dataset.add_noise,
        data_winow=cfg.dataset.data_window,
        label_window=cfg.dataset.label_window,
        time_shift=cfg.dataset.time_shift,
        total_wave_nums=cfg.dataset.total_wave_num
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_worklers
    )

    iterator = iter(dataloader)
    data, label = next(iterator)

    print(data.shape)
    print(label.shape)
    print(data)
    print(label)


# Test
if __name__ =='__main__':
    main()
