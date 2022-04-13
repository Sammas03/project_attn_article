import os
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class GruModel(AbsModel):
    '''
        进行gru神经网络预测的测试
    '''

    def __init__(self, config):
        super().__init__()
        self.predict_y = []
        self.real_y = []
        hidden_num = config['gru.hidden_num']
        output_num = config['output_size']
        num_layers = config['gru.num_layers']
        self.hidden_size = hidden_num
        self.lr = config['running.lr']
        self.input_size = config['input_size']
        self.layers = config['gru.num_layers']
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.layers,
                          # dropout=0.5
                          )

        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_num)
        )

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        x = x.permute(2, 0, 1)  # seq_len,batch,input_dim
        seq_len, batch, input_dim = x.shape
        h = init_rnn_hidden(batch=batch, hidden_size=self.hidden_size, num_layers=self.layers)
        y, h = self.gru(x, h)
        out = self.fc_out(h[-1, :, :])
        return out


class DesignGruCellModel(AbsModel):
    '''
         进行gru神经网络预测的测试
     '''

    def __init__(self, config):
        super().__init__()
        self.predict_y = []
        self.real_y = []
        hidden_num = config['gru.hidden_num']
        output_num = config['gru.output_num']
        num_layers = config['gru.num_layers']
        self.hidden_size = hidden_num
        self.lr = config['running.lr']
        self.input_size = config['input_size']
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.gru1 = nn.GRUCell(input_size=self.input_size, hidden_size=168)
        self.gru2 = nn.GRUCell(input_size=168, hidden_size=64)
        self.gru3 = nn.GRUCell(input_size=64, hidden_size=32)
        self.activation1 = nn.LeakyReLU()
        self.activation2 = nn.LeakyReLU()
        # self.gru2 = nn.GRU(input_size=hidden_num, hidden_size=32, num_layers=1)
        # self.activation2 = nn.LeakyReLU()
        # self.gru3 = nn.GRU(input_size=16, hidden_size=hidden_num, num_layers=1)
        self.output_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Linear(16, output_num)
        )

        # self.GRU_layer = nn.Sequential(
        #     nn.GRU(input_size=1, hidden_size=hidden_num, num_layers=num_layers),
        #     nn.Linear(hidden_num, output_num)
        # )

        self.hidden = None

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden

        x = x.permute(1, 0, 2)  # seq_len,batch,input_dim
        seq_len, batch, input_dim = x.shape
        h1, h2, h3 = init_rnn_hidden(batch, 168, 0), init_rnn_hidden(batch, 64, 0), init_rnn_hidden(batch, 32, 0)
        out = None
        for i in range(seq_len):
            h1 = self.gru1(x[i, :, :], h1)
            h2 = self.gru2(self.activation1(h1), h2)
            h3 = self.gru3(self.activation2(h2), h3)
            out = self.output_linear(h3)
        # _, h = self.gru1(x, h)
        # x1, h1 = self.gru1(x)
        # x1_act = self.activation1(x1)
        # x2, h2 = self.gru2(x1_act)
        # x2_act = self.activation2(x2)
        # x3, h3 = self.gru3(x2_act)
        #       out = self.output_linear(h[-1])  # 只是用最后一个隐藏层即可

        return out
