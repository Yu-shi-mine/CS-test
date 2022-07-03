"""
データローダーをリスト形式にまとめる
"""

from typing import List

import os, glob

from cv2 import transform
 
from torch.utils.data import Dataset, DataLoader

from datahandler.simple_dataset import SimpleDataset


def GenDataLoaderList(dataset_dir: str, phase: str, feature_length: int, input_feature_length, output_feature_length, sequence_length: int, batch_size: int, transform: object) -> List[DataLoader]:
    dataloaders = []
    data_folders = glob.glob(os.path.join(dataset_dir, phase, '*'))
    for folder in data_folders:
        csv_path = os.path.join(folder, 'joint/joint.csv')
        dataset = SimpleDataset(
            feature_length=feature_length,
            input_feature_length=input_feature_length,
            output_feature_length=output_feature_length,
            csv_path=csv_path,
            sequence_length=sequence_length,
            transform=transform
            )
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)
        dataloaders.append(dataloader)
    return dataloaders


if __name__ == '__main__':
    dataset_dir = './datasets/20220702_084913'
    phase = 'val'
    feature_length = 7
    input_feature_length = 6
    output_feature_length = 7
    sequence_length = 90
    batch_size = 1

    val_dataloader_list  = GenDataLoaderList(
        dataset_dir=dataset_dir,
        phase=phase,
        feature_length=feature_length,
        sequence_length=sequence_length,
        batch_size=batch_size
        )

    val_dataloader = val_dataloader_list[0]

    count = 0
    for x, y in val_dataloader:
        # print(f'x = {x}, y = {y}')
        print(x.shape)
        print(count)
        count += 1

