"""
データセットの作成
"""

from typing import Tuple

import os, glob

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class LstmDataset(Dataset):
    def __init__(self, csv_path_list: list, window_size: int, runup_length: int, inertia_length: int) -> None: 
        self.data_arr, self.label_arr = self.load_csv(csv_path_list, window_size, runup_length, inertia_length)

    def __len__(self) -> int:
        return self.data_arr.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data_arr[idx, :]
        x = torch.tensor(x)

        y = self.label_arr[idx, :]
        y = torch.tensor(y)
        return x, y
    
    def load_csv(self, csv_path_list: list, window_size: int, runup_length: int, inertia_length: int) -> Tuple[np.ndarray, np.ndarray]:
        data_list = []
        label_list = []
        for path in csv_path_list:
            # Load csv data
            raw_data: np.ndarray = np.loadtxt(path, delimiter=',', dtype='float32')
            if len(raw_data.shape) == 1:
                raw_data = raw_data[:, np.newaxis]
            elif len(raw_data.shape) == 2:
                pass
            else:
                raise Exception
            # print(raw_data.shape)
                
            # Extend raw data head and tail
            ex_data = self.runup_extend(data=raw_data, runup_length=runup_length)
            # print(ex_data.shape)
            ex_data = self.inertia_extend(data=ex_data, inertia_length=inertia_length)
            # print(ex_data.shape)

            # Split to windiwed data and label
            data, label = self.window_split(data=ex_data, window_size=window_size)
            # print(data.shape)
            # print(label.shape)
            data_list.append(data)
            label_list.append(label)

        data_arr = np.array(data_list)
        label_arr = np.array(label_list)
        # print(data_arr.shape)
        # print(label_arr.shape)
        data_arr = data_arr.reshape([-1, window_size, data_arr.shape[3]])
        label_arr = label_arr.reshape([-1, label_arr.shape[2]])
        # print(data_arr.shape)
        # print(label_arr.shape)

        return data_arr, label_arr
    
    def runup_extend(self, data: np.ndarray, runup_length: int) -> np.ndarray:
        if runup_length != 0:
            first_record = data[0, :]
            tiled_arr = np.tile(first_record, (runup_length, 1))
            extended_data = np.concatenate([tiled_arr, data], axis=0)
        else:
            extended_data = data
        return extended_data
    
    def inertia_extend(self, data: np.ndarray, inertia_length: int) -> np.ndarray:
        if inertia_length != 0:
            last_record = data[-1, :]
            tiled_arr = np.tile(last_record, (inertia_length, 1))
            extended_data = np.concatenate([data, tiled_arr], axis=0)
        else:
            extended_data = data
        return extended_data
    
    def window_split(self, data:np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        window_data = []
        window_label = []
        for i in range(0, data.shape[0]-window_size):
            x = data[i:i+window_size, :]
            y = data[i+window_size, :]
            window_data.append(x)
            window_label.append(y)
        windowed_data = np.array(window_data)
        windowed_label = np.array(window_label)
        return windowed_data, windowed_label


# Test
if __name__ == '__main__':
    # Setting
    dataset_dir = './datasets/02_20220710_213723_sincos'
    csv_path_list = glob.glob(os.path.join(dataset_dir, 'test/*/joint/joint.csv'))
    window_size = 360

    # Create Dataset
    joint_dataset = LstmDataset(
        csv_path_list=csv_path_list,
        window_size=window_size,
        runup_length=window_size,
        inertia_length=window_size
    )

    # Create DataLoader
    batch_size = 1
    dataloader = DataLoader(joint_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    # Operation check
    batch_iterator = iter(dataloader)
    x, y = next(batch_iterator)
    print('type(x) :', type(x))
    print('x.size() =', x.size())
    print('y.size() =', y.size())
    print('len(dataset) =', len(joint_dataset))
    # print('x = ', x)

