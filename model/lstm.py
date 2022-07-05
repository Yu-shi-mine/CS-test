"""
LSTMのモデル
"""

import numpy as np
from tensorboard import summary

import torch
from torch import nn
from torchinfo import summary


class LSTMSeq(nn.Module):
    def __init__(self, input_feature_length, output_feature_length) -> None:
        super().__init__()
        self.input_feature_length = input_feature_length
        self.output_feature_length = output_feature_length
        self.hidden_size = 80
        self.lstm = nn.LSTM(input_size=self.input_feature_length, hidden_size=self.hidden_size, batch_first=True, num_layers=1)
        self.fc = torch.nn.Linear(in_features=self.hidden_size, out_features=self.output_feature_length)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x, (h_n, c_n) = self.lstm(x)  # input should be shape: batch, length, input_size
        pred = self.fc(x)
        return pred


# Test
if __name__ == '__main__':
    input_feature_length = 1
    output_feature_length = 1
    net = LSTMSeq(input_feature_length=input_feature_length, output_feature_length=output_feature_length)
    print(summary(net, (4,10,1)))

