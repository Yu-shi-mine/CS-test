"""
教師データ等の生成&保存用
"""

import os, datetime

import numpy as np


# def gen_data(*args, div_ratio: int, noise_intensity: int) -> np.ndarray:
#     values_all = []
#     for x in range(0, 360, div_ratio):
#         values = []
#         for function in functions:
#             value = function(x)
#             noise = np.random.randint(-1, 1) * noise_intensity
#             value = value + noise
#             values.append(value)
#         values_all.append(values)
#     return np.array(values_all).reshape(len(values_all), len(values))


def gen_folders(data_dict: dict) -> None:
    for key, value in data_dict.items():
        dataset_key_path = os.path.join(dataset_dt, key)
        os.makedirs(dataset_key_path, exist_ok=True)
        for i in range(value):
            dataset_i_path = os.path.join(dataset_key_path, 'N{:0>2}'.format(i+1))
            os.makedirs(dataset_i_path, exist_ok=True)


# Test
if __name__ == '__main__':
    # Root folder setting
    dataset_root = "./datasets"
    os.makedirs(dataset_root, exist_ok=True)

    # Save folder
    dt = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    dataset_dt = os.path.join(dataset_root, dt)
    os.makedirs(dataset_dt, exist_ok=True)

    # Settings to generate datas
    data_dict = {'train': 1, 'val':1, 'test': 1}

    div_ratio = 1
    noise_intensity = 0.05

    # generate folders and datas
    for key, value in data_dict.items():
        dataset_key_path = os.path.join(dataset_dt, key)
        os.makedirs(dataset_key_path, exist_ok=True)
        for i in range(value):
            dataset_i_path = os.path.join(dataset_key_path, 'N{:0>2}'.format(i+1), 'joint')
            os.makedirs(dataset_i_path, exist_ok=True)
            t = np.arange(0, 360, div_ratio)
            arr_data = np.sin(np.pi * t / 180)
            noise = np.random.normal(loc=0, scale=0.1, size=arr_data.shape)
            arr_data += noise
            np.savetxt(os.path.join(dataset_i_path, 'joint.csv'), arr_data, delimiter=',')

    
    
    

