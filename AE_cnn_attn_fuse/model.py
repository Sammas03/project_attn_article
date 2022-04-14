import os
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class CnnAttnFuse(AbsModel):
    def __init__(self, config):
        super().__init__()
        self.lr = config['running.lr']
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.ar = Mlp_Uint(config,
                           input_size=72,
                           layer1=config['ar.layer1'],
                           layer2=int(config['ar.layer1']/2),
                           layer3=int(config['ar.layer1']/4),
                           out_size=1)
        if (config['gpu']):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def forward(self, x):
        # shape batch,col,history_len
        encoded = self.encoder(x)
        non_out = self.decoder(encoded, x[:, 0, :].unsqueeze(1))
        ar_out = self.ar(x[:,0,:])
        # 这里可以再接一个mlp作为ar
        return non_out + ar_out

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

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config['common.history_seq_len']
        in_attn_hidden_size = config['en.in_attn_hidden_size']
        self.in_attn = nn.Sequential(
            nn.Linear(input_size, in_attn_hidden_size),
            nn.Tanh(),
            nn.Linear(in_attn_hidden_size, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=7, padding=3)
        self.conv2_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=2, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2)
        self.conv4_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, dilation=2, padding=3)
        self.conv5_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=2, padding=4)

    def forward(self, x):
        # x input shape: batch,col,history_len
        e_t = self.in_attn(x)  # batch,col,1
        alpha_t = self.softmax(e_t)  # batch,col,1
        context_x = torch.bmm(alpha_t.permute(0, 2, 1), x)  # 实现将 x[col] 乘以 注意力并进行累加
        # 在多个conv1d下进行卷积，提取多个时间点的行为关联
        # conv1d卷积输出 batch,out_channel,seq
        seq1 = self.conv1(context_x)
        seq3 = self.conv3(context_x)
        seq5 = self.conv5(context_x)
        seq7 = self.conv7(context_x)
        seq2_2 = self.conv2_2(context_x)
        seq3_2 = self.conv3_2(context_x)
        seq4_2 = self.conv4_2(context_x)
        seq5_2 = self.conv5_2(context_x)
        # output shape:(batch,cnn_num,)
        return torch.cat([seq1, seq3, seq2_2, seq3_2, seq4_2], dim=1)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        temporal_attn_input_size = config['common.history_seq_len'] + 2 * config['de.lstm_hidden_size']
        self.lstm_hidden_size = config['de.lstm_hidden_size']
        lstm_input_size = config['common.history_seq_len'] + 1
        temporal_attn_hidden_size = config['de.temporal_attn_hidden_size']
        self.prediction_horizon = config['common.prediction_horizon']
        self.temporal_attn = nn.Sequential(
            nn.Linear(temporal_attn_input_size, temporal_attn_hidden_size),
            nn.Tanh(),
            nn.Linear(temporal_attn_hidden_size, 1)
        )

        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=1)
        self.fc_out = nn.Sequential(
            #nn.ReLU(),
            nn.Linear(self.lstm_hidden_size, int(self.lstm_hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.lstm_hidden_size / 2), 1)
            # nn.Linear(self.lstm_hidden_size, 1)
        )

    def forward(self, x: torch.Tensor, his: torch.Tensor):
        # shape batch,col,history_len
        self.batch_size = his.shape[0]
        channel = x.shape[1] # x batch,channel,history
        temp_p = his[:, :, -1]  # x 的初始维度为 (batch,col,history)
        output = torch.zeros((self.batch_size, self.prediction_horizon, 1), requires_grad=False)
        h_t = init_rnn_hidden(self.batch_size, self.lstm_hidden_size, num_layers=1)
        c_t = init_rnn_hidden(self.batch_size, self.lstm_hidden_size, num_layers=1)
        attn = torch.zeros(self.batch_size, self.prediction_horizon, channel)
        for t in range(self.prediction_horizon):
            x_hat = torch.cat([h_t.repeat(channel, 1, 1).permute(1, 0, 2),
                               c_t.repeat(channel, 1, 1).permute(1, 0, 2), x], dim=2)
            l_t = self.temporal_attn(x_hat)
            b_t = self.softmax(l_t)
            context = torch.bmm(b_t.permute(0, 2, 1), x)
            context_hat = torch.cat([context, temp_p.unsqueeze(1)], dim=2).permute(1, 0, 2)
            decode, (h_t, c_t) = self.lstm(context_hat, (h_t, c_t))
            out = self.fc_out(decode.permute(1, 0, 2))
            temp_p = out.view(self.batch_size, -1)
            output[:, t, :] = out.view(self.batch_size, -1)
            attn[:, t, :] = b_t.view(self.batch_size, -1)
        return output.squeeze(2)  # 最后一维默认为1，直接压缩掉
