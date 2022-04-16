import os
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


# 首先对输入的矩阵提取特征  3，1。 22 32 42 52

class CnnFilter(AbsModel):
    # 这里涉及到一个问题，先输出再融合还是先融合再输出
    def __init__(self, config):
        super().__init__()
        self.lr = config['running.lr']
        self.num_dir = 2 if config['de.bidirectional'] else 1
        self.fc_out_hidden_size =5 * config['de.lstm_hidden_size'] * self.num_dir # 系数为解码器个数

        '''multi header encoder'''
        self.hen1 = HeaderEncoder(config)
        self.hen2 = HeaderEncoder(config)
        self.hen3 = HeaderEncoder(config)
        self.hen4 = HeaderEncoder(config)
        '''multi header decoder'''
        self.hde1 = HeaderDecoder(config)
        self.hde2 = HeaderDecoder(config)
        self.hde3 = HeaderDecoder(config)
        self.hde4 = HeaderDecoder(config)
        self.hde5 = HeaderDecoder(config)

        self.ar = Mlp_Uint(config,
                           input_size=72,
                           layer1=config['ar.layer1'],
                           layer2=int(config['ar.layer1'] / 2),
                           layer3=int(config['ar.layer1'] / 4),
                           out_size=1)
        self.fc_out = nn.Sequential(
            nn.Linear(self.fc_out_hidden_size, int(self.fc_out_hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.fc_out_hidden_size / 2), int(self.fc_out_hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.fc_out_hidden_size / 4), 1)
        )

        if (config['gpu']):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def forward(self, x):
        # shape batch,col,history_len
        """encoder"""
        hout_1 = self.hen1(x[:, 0, :].unsqueeze(1))
        hout_2 = self.hen1(x[:, 1, :].unsqueeze(1))
        hout_3 = self.hen1(x[:, 2, :].unsqueeze(1))
        hout_4 = self.hen1(x[:, 3, :].unsqueeze(1))
        conv_cat1 = self.cat_same_conv_tensor(0, hout_1, hout_2, hout_3, hout_4)
        conv_cat2 = self.cat_same_conv_tensor(1, hout_1, hout_2, hout_3, hout_4)
        conv_cat3 = self.cat_same_conv_tensor(2, hout_1, hout_2, hout_3, hout_4)
        conv_cat4 = self.cat_same_conv_tensor(3, hout_1, hout_2, hout_3, hout_4)
        conv_cat5 = self.cat_same_conv_tensor(4, hout_1, hout_2, hout_3, hout_4)
        '''decoder'''
        dout1 = self.hde1(conv_cat1)
        dout2 = self.hde1(conv_cat2)
        dout3 = self.hde1(conv_cat3)
        dout4 = self.hde1(conv_cat4)
        dout5 = self.hde1(conv_cat5)
        dout_sum = torch.cat((dout1, dout2, dout3, dout4, dout5),dim=1)
        non_out = self.fc_out(dout_sum) # 非线性输出
        #ar_out = self.ar(x[:, 0, :])
        # 这里可以再接一个mlp作为ar
        return non_out # + ar_out

    def cat_same_conv_tensor(self, conv_no, *header):
        summary = []
        for conv_out in header:
            summary.append(conv_out[:, conv_no, :].unsqueeze(1))
        return torch.cat(summary, dim=1)


class Mlp_Uint(nn.Module):
    '''
         进行gru神经网络预测的测试
     '''

    def __init__(self, config, input_size=None, layer1=None, layer2=None, layer3=None, out_size=None):
        super().__init__()
        self.config = config
        input_size = input_size if input_size else config['common.history_seq_len']
        hidden_num_1 = layer1 if layer1 else config['unit.layer1.hidden_num']
        hidden_num_2 = layer2 if layer2 else config['unit.layer2.hidden_num']
        hidden_num_3 = layer3 if layer3 else config['unit.layer3.hidden_num']
        output_num = out_size if out_size else config['aemlp.encode_size']
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_num_1),
            nn.Tanh(),
            nn.Linear(hidden_num_1, hidden_num_2),
            nn.Tanh(),
            nn.Linear(hidden_num_2, hidden_num_3),
            nn.Tanh(),
            nn.Linear(hidden_num_3, output_num)
        )

    def forward(self, x: torch.Tensor):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # x原始维度，batch,seq_len,input_dim ，input_dim==1
        out = self.mlp(x)
        return out


class HeaderEncoder(nn.Module):
    def __init__(self, config):
        super(HeaderEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=2, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2)
        self.conv4_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, dilation=2, padding=3)

    def forward(self, x):
        seq1 = self.conv1(x)
        seq3 = self.conv3(x)
        seq2_2 = self.conv2_2(x)
        seq3_2 = self.conv3_2(x)
        seq4_2 = self.conv4_2(x)
        return torch.cat([seq1, seq3, seq2_2, seq3_2, seq4_2], dim=1)


class HeaderDecoder(nn.Module):
    def __init__(self, config):
        super(HeaderDecoder, self).__init__()
        self.lstm_hidden_size = config['de.lstm_hidden_size']
        lstm_input_size = 4 # 与encoder 数量一致 ，每个encoder产生一种卷积之后的数据
        self.prediction_horizon = config['common.prediction_horizon']
        self.num_layer = config['de.lstm_num_layer']
        self.num_dir = 2 if config['de.bidirectional'] else 1
        attn_input_size = config['common.history_seq_len']
        in_attn_hidden_size = config['in_attn_hidden_size']
        self.in_attn = nn.Sequential(
            nn.Linear(attn_input_size, in_attn_hidden_size),
            nn.Tanh(),
            nn.Linear(in_attn_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                             hidden_size=self.lstm_hidden_size,
                             num_layers=self.num_layer,
                             bidirectional=config['de.bidirectional'],
                             dropout=config['de.dropout'])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        self.batch_size = x.shape[0]
        e_t = self.in_attn(x)  # batch,col,1
        alpha_t = self.softmax(e_t)  # batch,col,1
        x_hat = torch.mul(x, alpha_t) # 进行加权
        x_hat = x_hat.permute(2, 0, 1)  # shape: batch,col,history ->history,batch,col
        h_t = init_rnn_hidden(self.batch_size, self.lstm_hidden_size, num_layers=self.num_layer, num_dir=self.num_dir)
        c_t = init_rnn_hidden(self.batch_size, self.lstm_hidden_size, num_layers=self.num_layer, num_dir=self.num_dir)
        x_out, (h_t, c_t) = self.lstm(x_hat, (h_t, c_t))  # shape batch,col,history_len
        return x_out[-1, :, :]

