"""
データセットの作成
"""

from typing import Tuple

import os, glob

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class JointDataset(Dataset):
    def __init__(self, feature_length, dataset_dir, phase, sequence_length, transform=None) -> None:
        self.feature_length = feature_length
        self.raw_data = self.gen_raw_data(dataset_dir, phase)  # -> np.array
        self.sequence_length = sequence_length
        self.try_times = self.raw_data.shape[0]
        self.total_time_length = self.raw_data.shape[1]
        self.transform = transform
        if self.transform is not None:
            self.raw_data = transform(self.raw_data)

    def gen_raw_data(self, dataset_dir, phase) -> np.ndarray:
        try_time_folders = glob.glob(os.path.join(dataset_dir, phase, '*'))
        csv_arr_list = []
        for folder in try_time_folders:
            # joint
            csv_arr: np.ndarray = np.loadtxt(os.path.join(folder, 'joint/joint.csv'), delimiter=',', dtype='float32')
            csv_arr = csv_arr.reshape([csv_arr.shape[0], self.feature_length])
            csv_arr_list.append(csv_arr)
        raw_data = np.array(csv_arr_list)
        raw_data = raw_data.reshape([len(try_time_folders), csv_arr.shape[0], self.feature_length])

        return raw_data

    def __len__(self) -> int:
        return (self.total_time_length - self.sequence_length + 1) * self.try_times

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        try_time, index = self.get_time_and_index(idx)

        x = self.raw_data[try_time, index : index+self.sequence_length, :]
        x = torch.tensor(x)

        y = self.raw_data[try_time, index+1 : index+self.sequence_length+1, :]
        y = torch.tensor(y)
        return x, y
    
    def get_time_and_index(self, idx) -> int:
        try_time = (idx) // (self.total_time_length - self.sequence_length + 1) 
        index = idx - try_time * (self.total_time_length - self.sequence_length + 1)
        return try_time, index


# Test
if __name__ == '__main__':
    dataset_dir = './datasets/20220702_084913'
    phase = 'train'
    feature_length = 1
    sequence_length = 2
    batch_size = 1
    transform = None

    joint_dataset = JointDataset(
        feature_length=feature_length,
        dataset_dir=dataset_dir, 
        phase=phase, 
        sequence_length=sequence_length
        )

    dataloader = DataLoader(joint_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    count=0
    for x, y in dataloader:
        # print(f'x = {x}, y = {y}')
        # print(x.shape)
        print(count)
        count += 1
        

