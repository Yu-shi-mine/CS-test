"""
データセットの作成
"""

from typing import Tuple

import os, glob

from cv2 import demosaicing

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from utils.transform import MinMaxScaler


class SimpleDataset(Dataset):
    def __init__(self, feature_length: int, input_feature_length: int, output_feature_length: int, csv_path: str, sequence_length: int, transform: object=None) -> None:
        self.feature_length = feature_length
        self.input_feature_length = input_feature_length
        self.output_feature_length = output_feature_length
        self.raw_data = np.loadtxt(csv_path, delimiter=',', dtype='float32')
        self.raw_data = self.raw_data.reshape([-1, self.feature_length])
        self.sequence_length = sequence_length
        self.total_time_length = self.raw_data.shape[0]
        if transform is not None:
            self.raw_data, _ = transform(self.raw_data)

    def __len__(self) -> int:
        return self.total_time_length - self.sequence_length - 1

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.raw_data[idx : idx+self.sequence_length, :self.input_feature_length]
        x = torch.tensor(x)

        y = self.raw_data[idx+1 : idx+self.sequence_length+1, :self.output_feature_length]
        y = torch.tensor(y)

        return x, y


# Test
if __name__ == '__main__':
    csv_path = './datasets/20220702_084913/train/N02/joint/joint.csv'
    feature_length = 7
    input_feature_length = 6
    output_feature_length = 7
    sequence_length = 20
    batch_size = 2
    transform = MinMaxScaler(feature_length=feature_length)

    joint_dataset = SimpleDataset(
        feature_length=feature_length,
        input_feature_length=input_feature_length,
        output_feature_length=output_feature_length,
        csv_path=csv_path, 
        sequence_length=sequence_length
        )

    dataloader = DataLoader(joint_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    count=0
    for x, y in dataloader:
        # print(f'x = {x}, y = {y}')
        print(x.shape)
        print(count)
        count += 1