"""
推論用
"""

import os, glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datahandler.lstm_dataset import LstmDataset
from model.lstm import LSTMSeq

def infer(net: nn.Module, dataloader: DataLoader):
    # NOTE: You'd better to set batch size to '1' in inference mode
    # NOTE: input_shape and output shape should be equal
    with torch.no_grad():
        net.eval()

        isFirst = True
        for x, y in dataloader:
            output = net(x)
            pred = output.numpy()   # shape: batch, output_size
            label = y.numpy()       # shape: batch, output_size

            if isFirst:
                pred_arr = pred
                label_arr = label
                isFirst = False
            else:
                pred_arr = np.concatenate([pred_arr, pred], axis=0)
                label_arr = np.concatenate([label_arr, label], axis=0)

        return pred_arr, label_arr

def recursive_infer(net: nn.Module, dataloader: DataLoader):
    # NOTE: You'd better to set batch size to '1' in inference mode
    # NOTE: input_shape and output shape should be equal
    with torch.no_grad():
        net.eval()

        isFirst = True
        for x, y in dataloader:
            if isFirst:
                input = x   # input shape: batch, length, input_size

            output = net(input) # output shape: batch, output_size
            
            pred = output.numpy()   # shape: batch, output_size
            label = y.numpy()       # shape: batch, output_size

            if isFirst:
                pred_arr = pred
                label_arr = label
                isFirst = False
            else:
                pred_arr = np.concatenate([pred_arr, pred], axis=0)
                label_arr = np.concatenate([label_arr, label], axis=0)

            # Prepare for next step
            previous = input[:, 1:, :]      # shape: batch, window_size-1, input_size
            next_head = output.unsqueeze(1) # shape: batch, 1, output_size
            input = torch.cat([previous, next_head], dim=1) # shape: batch, window_size, input_size

        return pred_arr, label_arr

def sort_results(pred_arr: np.ndarray, label_arr: np.ndarray) -> pd.DataFrame:
    for i in range(pred_arr.shape[1]):
        pred_col = pred_arr[:, i].reshape([-1, 1])
        label_col = label_arr[:, i].reshape([-1, 1])
        result_col = np.concatenate([pred_col, label_col], axis=1)
        if i == 0:
            result = result_col
        else:
            result = np.concatenate([result, result_col], axis=1)
    return result


# Test
if __name__ == '__main__':
    # Settings
    RECURSIVE = True
    DATASET_ROOT = './datasets/02_20220710_213723_sincos'
    TRAINED_WEIGHT_PATH = './log/20220710_220609/weights_100.pth'

    result_dir = './result/'
    result_dir = os.path.join(result_dir, datetime.today().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(result_dir, exist_ok=True)

    #   Create Dataset
    test_path_list = glob.glob(os.path.join(DATASET_ROOT, 'test/*/joint/joint.csv'))
    
    window_size = 10
    runup_length  = 10
    inertia_length = 10

    test_datast = LstmDataset(
        csv_path_list=test_path_list,
        window_size=window_size,
        runup_length= runup_length,
        inertia_length=inertia_length
    )

    # Create DataLoader
    # NOTE: You'd better to set batch size to '1' in inference mode
    batch_size = 1
    test_dataloader = DataLoader(
        dataset=test_datast,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False
    )

    # Define network
    input_size = 2
    lstm_input = 100
    hidden_size = 200
    output_size = 2

    net = LSTMSeq(
        input_size=input_size,
        lstm_input=lstm_input,
        hidden_size=hidden_size,
        output_size=output_size
    )

    net_weights = torch.load(TRAINED_WEIGHT_PATH, map_location={'cuda:0':'cpu'})
    net.load_state_dict(net_weights)

    # Infer
    if not RECURSIVE:
        pred_arr, label_arr = infer(net=net, dataloader=test_dataloader)
        # print(pred_arr.shape)
        # print(label_arr.shape)
    else:
        pred_arr, label_arr = recursive_infer(net=net, dataloader=test_dataloader)

    # Save result
    result_arr = sort_results(pred_arr=pred_arr, label_arr=label_arr)
    col_name = ['pred_sin', 'truth_sin', 'pred_cos', 'truth_cos', 'pred_cubic', 'truth_cubic']
    df = pd.DataFrame(result_arr, columns=col_name)

    df.to_csv(os.path.join(result_dir, 'result.csv'))
