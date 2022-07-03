"""
データの前処理
"""

from typing import Tuple

import os

import numpy as np


class MinMaxScaler():
    def __init__(self, feature_length: int) -> None:
        self.feature_length = feature_length

    def __call__(self, data: np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
        return self.raw_to_transformed(data=data)

    def raw_to_transformed(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.min_max = []
        for i in range(self.feature_length):
            sliced_data = data[:, i]
            _max = np.max(sliced_data)
            _min = np.min(sliced_data)
            transformed_data = (sliced_data - _min)/(_max - _min)
            transformed_data = transformed_data.reshape([-1, 1])
            if i == 0:
                return_array = transformed_data
            else:
                return_array = np.concatenate([return_array, transformed_data], axis=1)
            self.min_max.append([_min, _max])
        self.min_max = np.array(self.min_max)
        self.min_max = self.min_max.reshape([self.feature_length, 2])

        return return_array, self.min_max

    
    def transformed_to_raw(self, data:np.ndarray) -> np.ndarray:
        for i in range(self.feature_length):
            sliced_data = data[:, i]
            _min = self.min_max[i, 0]
            _max = self.min_max[i, 1]
            inversed_data = sliced_data * (_max - _min) + _min
            inversed_data = inversed_data.reshape([-1, 1])
            if i == 0:
                return_array = inversed_data
            else:
                return_array = np.concatenate([return_array, inversed_data], axis=1)
        return return_array


# Test
if __name__ == '__main__':
    feature_length = 7
    input_feature_length = 6
    output_feature_length = 7
    dataset_path = './datasets/04_20220703_175113_sincoslinsqu_train30_val10_test1/test/N01/joint'
    csv_path = os.path.join(dataset_path, 'joint.csv')

    min_max_scaler = MinMaxScaler(feature_length=feature_length)
    data = np.loadtxt(csv_path, delimiter=',', dtype='float32')

    transformed_data, min_max = min_max_scaler.raw_to_transformed(data=data)
    np.savetxt(os.path.join(dataset_path, 'transformed_joint.csv'), transformed_data, delimiter=',')
    np.savetxt(os.path.join(dataset_path, 'min_max.csv'), min_max, delimiter=',')

    inversed_data = min_max_scaler.transformed_to_raw(data=transformed_data, min_max=min_max)
    np.savetxt(os.path.join(dataset_path, 'inversed_data.csv'), inversed_data, delimiter=',')


