"""
推論用
"""

import os, glob
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader_list import GenDataLoaderList
from utils.transform import MinMaxScaler
from model.lstm import LSTMSeq

def infer(net: nn.Module, weight_path: str, dataloader_dict: dict, result_dir: str, sequence_length: int):
    with torch.no_grad():
        net_weights = torch.load(weight_path, map_location={'cuda:0':'cpu'})
        net.load_state_dict(net_weights)
        net.eval()
        results = []
        for x, y in dataloader_dict['test'][0]:
            pred = net(x).numpy()
            y = y.numpy()
            pred = pred[0, sequence_length-1, :]
            y = y[0, sequence_length-1, :]
            result = np.concatenate([pred, y])
            results.append(result)
        results = np.array(results)
        np.savetxt(os.path.join(result_dir, 'result.csv'), results, delimiter=',')

def recursive_infer(
    net: nn.Module, weight_path: str, input_path: str,
    result_dir: str, sequence_length: int, feature_length: int, 
    input_feature_length, transform: MinMaxScaler):

    with torch.no_grad():
            net_weights = torch.load(weight_path, map_location={'cuda:0':'cpu'})
            net.load_state_dict(net_weights)
            net.eval()
            results = []
            raw_data = np.loadtxt(input_path, delimiter=',', dtype='float32')
            raw_data, _ = transform.raw_to_transformed(raw_data)
            count = raw_data.shape[0] - sequence_length - 1

            for i in range(count):
                if i == 0:
                    x = raw_data[:, :input_feature_length]
                    x = x.reshape([1, raw_data.shape[0], input_feature_length])
                    x = x[:, :sequence_length, :]
                    x = torch.from_numpy(x)
                    pred = net(x)
                else:
                    pred = net(x)

                pred_arr = pred.numpy()
                pred_arr_inv = transform.transformed_to_raw(pred_arr.reshape([-1, feature_length]))
                pred_arr_inv = pred_arr_inv[-1, :].reshape([1, 1, feature_length])
                y = raw_data.reshape([1, raw_data.shape[0], feature_length])
                y = y[:, i+ sequence_length+1, :].reshape([1, 1, feature_length])
                result = np.concatenate([pred_arr_inv, y], axis=2)
                results.append(result)
                
                x_arr  = x.numpy()
                x_arr = np.delete(x_arr, obj=0, axis=1)
                pred_arr = pred_arr[:, -1, :input_feature_length].reshape([1, 1, input_feature_length])
                x = np.concatenate([x_arr, pred_arr], axis=1)
                x = torch.from_numpy(x)

            results = np.array(results)
            results = results.reshape([count, feature_length * 2])
            np.savetxt(os.path.join(result_dir, 'result.csv'), results, delimiter=',')


# Test
if __name__ == '__main__':
    # Settings
    dataset_dir = './datasets/06_20220703_213905_7feature_train30_val10_test1'
    feature_length = 7
    input_feature_length = 6
    output_feature_length = 7
    sequence_length = 90
    batch_size = 1
    transform = None
    weight_path = './log/20220703_215034_dataset06_seq90/weights_100.pth'
    result_dir = './result/'
    result_dir = os.path.join(result_dir, datetime.today().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(result_dir, exist_ok=True)
    transform = MinMaxScaler(feature_length=feature_length)

    #   Create DataLoader List
    phase = 'test'
    test_dataloader_list = GenDataLoaderList(
        dataset_dir=dataset_dir,
        phase=phase,
        feature_length=feature_length,
        input_feature_length=input_feature_length,
        output_feature_length=output_feature_length,
        sequence_length=sequence_length,
        batch_size=batch_size,
        transform=transform
        )

    dataloader_dict = {'test': test_dataloader_list}

    # Define network
    net = LSTMSeq(input_feature_length=input_feature_length, output_feature_length=output_feature_length)

    # infer
    # infer(
    #     net=net,
    #     weight_path=weight_path,
    #     dataloader_dict=dataloader_dict,
    #     result_dir=result_dir,
    #     seqence_length=sequence_length
    # )

    # Reccursive inferece
    input_path = os.path.join(dataset_dir, 'test/N01/joint/joint.csv')
    recursive_infer(
        net=net,
        weight_path=weight_path,
        input_path=input_path,
        result_dir=result_dir,
        sequence_length=sequence_length,
        feature_length=feature_length,
        input_feature_length=input_feature_length,
        transform=transform
    )

