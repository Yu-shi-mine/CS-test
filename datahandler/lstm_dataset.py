"""
データセットの作成
"""

from typing import Tuple

import os, glob

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class LstmDataset(Dataset):
    def __init__(self, window_size: int, csv_path_list: list, time_shift: int) -> None:
        self.window_size = window_size
        self.data_arr, self.label_arr = self.load_csv(csv_path_list, time_shift)

    def __len__(self) -> int:
        return self.data_arr.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data_arr[idx, :]
        x = torch.tensor(x)

        y = self.label_arr[idx, :]
        y = torch.tensor(y)
        return x, y
    
    def load_csv(self, csv_path_list: list, time_shift: int) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        label = []
        for i, path in enumerate(csv_path_list):
            raw_data = np.loadtxt(path, delimiter=',', dtype='float32')
            extended_data = self.runup_extend(data=raw_data)
            windowed_data, windowed_label = self.window_split(data=extended_data, time_shift=time_shift)
            data.append(windowed_data)
            label.append(windowed_label)
        _data_arr = np.array(data)
        _label_arr = np.array(label)
        _data_arr = _data_arr.reshape([-1, self.window_size, _data_arr.shape[3]])
        _label_arr = _label_arr.reshape([-1, self.window_size, _label_arr.shape[3]])

        return _data_arr, _label_arr
    
    def runup_extend(self, data:np.ndarray) -> np.ndarray:
        data = data.reshape([data.shape[0], -1])
        tiled_arr = np.tile(data[0, :], [self.window_size, data.shape[1]])
        extended_data = np.concatenate([tiled_arr, data], axis=0)
        return extended_data
    
    def window_split(self, data:np.ndarray, time_shift: int) -> Tuple[np.ndarray, np.ndarray]:
        window_data = []
        window_label = []
        for i in range(data.shape[0] - self.window_size - time_shift):
            x = data[i:i+self.window_size, :]
            y = data[i+time_shift:i+self.window_size+time_shift, :]
            window_data.append(x)
            window_label.append(y)
        _windowed_data = np.array(window_data)
        _windowed_label = np.array(window_label)
        return _windowed_data, _windowed_label


# Test
if __name__ == '__main__':
    # Setting
    dataset_dir = './datasets/07_20220705_183400_sin'
    csv_path_list = glob.glob(os.path.join(dataset_dir, 'test/*/joint/joint.csv'))
    windoe_size = 180

    # Create Dataset
    joint_dataset = LstmDataset(
        window_size=windoe_size,
        csv_path_list=csv_path_list,
        time_shift = 1
        )

    # Create DataLoader
    batch_size = 32
    dataloader = DataLoader(joint_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    # Operation check
    batch_iterator = iter(dataloader)
    x, y = next(batch_iterator)
    print('type(x) :', type(x))
    print('x.size() =', x.size())
    print('len(dataset) =', len(joint_dataset))
    # print('x = ', x)

