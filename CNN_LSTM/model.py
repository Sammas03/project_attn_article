import os
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class CnnLstmModel(AbsModel):
    '''
        进行lstm神经网络预测的测试
    '''

    def __init__(self, config):
        super().__init__()
        self.predict_y = []
        self.real_y = []

        hidden_num = config['lstm.hidden_num']
        output_num = config['output_size']
        self.hidden_size = hidden_num
        self.lr = config['running.lr']
        self.input_size = config['input_size']
        self.layers = config['lstm.num_layers']
        self.conv = nn.Conv1d(in_channels=4, out_channels=self.input_size,kernel_size=3,padding=1)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.layers,
                            # dropout=0.5
                            )
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_num)
        )
        # self.weight_init()

    def forward(self, x):
        x = self.conv(x)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        x = x.permute(2, 0, 1)  # seq_len,batch,input_dim
        seq_len, batch, input_dim = x.shape
        # h = init_rnn_hidden(batch=batch, hidden_size=self.hidden_size, num_layers=self.layers)
        # c = init_rnn_hidden(batch=batch, hidden_size=self.hidden_size, num_layers=self.layers)
        y, (h, c) = self.lstm(x)
        out = self.fc_out(h[-1, :, :])
        return out