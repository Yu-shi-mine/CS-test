"""
教師データ等の生成&保存用
"""

import os, datetime

import numpy as np
from torch import div
from utils.functions import AsinX, AcosX, LinearFunc, SquareFunc, CubicFunc, MyFunc, MyFunc2


def gen_data(*args, div_ratio: int, noise_intensity: int) -> np.ndarray:
    values_all = []
    for x in range(0, 360, div_ratio):
        values = []
        for function in functions:
            value = function(x)
            noise = np.random.randint(-1, 1) * noise_intensity
            value = value + noise
            values.append(value)
        values_all.append(values)
    return np.array(values_all).reshape(len(values_all), len(values))


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
    data_dict = {'train': 30, 'val':10, 'test': 1}

    # Funcions
    sin = AsinX(a=1)
    cos = AcosX(a=0.4)
    linear = LinearFunc(a=2, b=3)
    square = SquareFunc(a=-0.3, b=4, c=-1.2)
    cubic = CubicFunc(a=0.7, b=-4, c=2, d=0.6)
    myfunc = MyFunc(a=0.4, b=-0.6, c=0.4)
    myfunc2 = MyFunc2(a=0.4, b=0.6, c=0.7)
    functions = [sin]

    div_ratio = 1
    noise_intensity = 0.05

    # generate folders and datas
    for key, value in data_dict.items():
        dataset_key_path = os.path.join(dataset_dt, key)
        os.makedirs(dataset_key_path, exist_ok=True)
        for i in range(value):
            dataset_i_path = os.path.join(dataset_key_path, 'N{:0>2}'.format(i+1), 'joint')
            os.makedirs(dataset_i_path, exist_ok=True)
            arr_data = gen_data(functions, div_ratio=div_ratio, noise_intensity=noise_intensity)
            np.savetxt(os.path.join(dataset_i_path, 'joint.csv'), arr_data, delimiter=',')

    
    
    

