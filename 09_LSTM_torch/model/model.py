"""
LSTM model
"""

from typing import Tuple

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn


class LSTMseq(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(LSTMseq, self).__init__()
        self.lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, hn, cn) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        x, (hn_1, cn_1) = self.lstm_layer(x, (hn.detach(), cn.detach()))
        out_1 = self.output_layer(x)

        x, (hn_2, cn_2) = self.lstm_layer(out_1, (hn_1, cn_1))
        out_2 = self.output_layer(x)

        x, (hn_3, cn_3) = self.lstm_layer(out_2, (hn_2, cn_2))
        out_3 = self.output_layer(x)

        x, (hn_4, cn_4) = self.lstm_layer(out_3, (hn_3, cn_3))
        out_4 = self.output_layer(x)

        x, (hn_5, cn_5) = self.lstm_layer(out_4, (hn_4, cn_4))
        out_5 = self.output_layer(x)

        outputs = torch.cat([out_1, out_2, out_3, out_4, out_5], dim=1)
        return outputs, hn_1, cn_1

class LSTMStateful(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(LSTMStateful, self).__init__()
        self.lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x, hn, cn) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        x, (hn, cn) = self.lstm_layer(x, (hn.detach(), cn.detach()))
        x = self.output_layer(x)
        return x, hn, cn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(SimpleLSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        x, (hn, cn) = self.lstm_layer(x)
        x = self.output_layer(x)
        return x


@hydra.main(config_name='config', config_path='../config')
def main(cfg: DictConfig):
    model = SimpleLSTM(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        output_size=cfg.model.output_size
    )

    x = torch.rand(size=[cfg.dataloader.batch_size, cfg.dataset.data_window, cfg.model.input_size])
    hn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])
    cn = torch.zeros(size=[1, cfg.dataloader.batch_size, cfg.model.hidden_size])

    output = model(x)

    print(output.shape)
    # print(hn_1.shape)
    # print(cn_1.shape)


# Test
if __name__ == '__main__':
    main()
