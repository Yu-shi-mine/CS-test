"""
LSTMのモデル
"""

import numpy as np
import pandas as pd

import torch
from torch import nn
from torchinfo import summary


class LSTMSeq(nn.Module):
    def __init__(self, lstm_input: int, input_size: int, hidden_size: int, output_size: int) -> None:
        super(LSTMSeq, self).__init__()
        self.input_size = input_size
        self.lstm_input = lstm_input
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_layer = nn.Linear(in_features=self.input_size, out_features=self.lstm_input)
        self.lstm = nn.LSTM(input_size=self.lstm_input, hidden_size=self.hidden_size, batch_first=True, num_layers=1)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Fullly connected layer
        x = self.input_layer(x)

        # LSTM layer
        x, (h_n, c_n) = self.lstm(x)    # input should be shape: batch, length, input_size
                                        # output of lstm layer : batch, length, hidden_size
                                        # h_n and c_n are shape: 1, batch, input_size
        
        # slice LSTM output
        x = x[:, -1, :]     # Many To One Task only needs last time step of lstm layer output.
                            # '-1' means the last time step.
                            # If num_layer=1, x[:, -1, :] is equal to h_n[0].

        # Sigmodi layer
        # x = self.sigmoid(x)

        # Fully Connected layer
        pred = self.fc(x)   # output shape: batch, output_size

        return pred


# Test
if __name__ == '__main__':
    input_size = 4
    lstm_input = 40
    hidden_size = 80
    output_size = 4
    
    net = LSTMSeq(
        input_size=input_size,
        lstm_input=lstm_input,
        hidden_size=hidden_size,
        output_size=output_size
    )
    
    print(summary(net, (4, 10, 4)))


