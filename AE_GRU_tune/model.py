import os
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class AeGruModel(AbsModel):
    def __init__(self, config):
        super(AeGruModel, self).__init__()
        self.config = config
        self.lr = config['running.lr']  # configure_optimizers使用
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        return out


class GruUnit(nn.Module):
    '''
          进行gru神经网络预测的测试
      '''

    def __init__(self, input_size, hidden_num, layers):
        super().__init__()
        self.layers = layers
        self.hidden_num = hidden_num
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_num,
                          num_layers=layers,
                          # dropout=0.5
                          )

    def forward(self, x):
        seq_len, batch, input_dim = x.shape
        h = init_rnn_hidden(batch=batch, hidden_size=self.hidden_num, num_layers=self.layers)
        y, h = self.gru(x, h)  # y (L, N, D * H_{out})
        return y, h


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        hidden_num = config['en.hidden_num']
        input_size = config['input_size']
        layers = config['en.num_layers']
        self.encoder = GruUnit(input_size=input_size, hidden_num=hidden_num, layers=layers)

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_num)
        x = x.permute(2, 0, 1)  # seq_len,batch,input_dim
        return self.encoder(x)[0]


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_num = config['en.hidden_num']
        input_size = config['en.hidden_num']
        layers = config['en.num_layers']
        out_size = config['output_size']
        self.decoder = GruUnit(input_size=input_size, hidden_num=hidden_num, layers=layers)
        self.fc_out = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(hidden_num, out_size)
        )

    def forward(self, x):
        y, h = self.decoder(x)
        return self.fc_out(h[-1, :, :])
