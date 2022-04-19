import os
from _ast import In

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class MainModel(AbsModel):
    '''
        进行gru神经网络预测的测试
    '''

    def __init__(self, config):
        super().__init__()
        self.lr = config['running.lr']  # configure_optimizers使用
        self.main_encoder = MainEncoder(config)
        self.sup_encoder = SupEncoder(config)
        self.decoder = TemproalDecoder(config)
        self.ar = Mlp_Uint(config,
                           input_size=config['common.history_seq_len'],
                           layer1=config['ar.layer1'],
                           layer2=int(config['ar.layer1'] / 2),
                           layer3=int(config['ar.layer1'] / 4),
                           out_size=1)
        # 初始化和cuda
        self.weight_init()
        if (config['gpu']):
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        # encoded, attn = self.encoder(x)
        # output = self.decoder(encoded, x)
        # return output
        # 相比较源模型 ，做出了变形将原始数据分离
        main_seq = x[:, 0, :].unsqueeze(1)
        sup_seq = (x[:, 1:, -24:] - x[:, 1:, -25:-1])  # 辅助数据做一阶差分
        main_code, attn = self.main_encoder(main_seq)
        sup_code = self.sup_encoder(sup_seq)
        nonout = self.decoder(main_code, sup_code)  # main_seq
        ar_out = self.ar(x[:, 0, :])
        fout = self.fc(torch.cat([nonout, ar_out], dim=1))
        # p=0.7
        # fout = p*nonout + (1-p)* ar_out
        return fout


class Mlp_Uint(nn.Module):
    '''
         进行gru神经网络预测的测试
     '''

    def __init__(self, config, input_size=None, layer1=None, layer2=None, layer3=None, out_size=None):
        super().__init__()
        self.config = config
        self.dropout = config['ar.dropout']
        input_size = input_size if input_size else config['common.history_seq_len']
        hidden_num_1 = layer1 if layer1 else config['unit.layer1.hidden_num']
        hidden_num_2 = layer2 if layer2 else config['unit.layer2.hidden_num']
        hidden_num_3 = layer3 if layer3 else config['unit.layer3.hidden_num']
        output_num = out_size if out_size else config['aemlp.encode_size']
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_num_1),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(hidden_num_1, hidden_num_2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(hidden_num_2, output_num),
            # nn.ReLU(),
            # nn.Linear(hidden_num_3, output_num)
        )

    def forward(self, x: torch.Tensor):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # x原始维度，batch,seq_len,input_dim ，input_dim==1
        out = self.mlp(x)
        return out


class MainEncoder(nn.Module):

    def __init__(self, config):
        super(MainEncoder, self).__init__()
        self.batch_size = config['running.batch_size']
        self.block_num = config['common.block_num']
        self.block_len = config['common.block_len']
        self.hidden_size = config['en.hidden_size']
        self.out_channels = config['en.out_channels']
        self.lstm_num_layers = config['en.num_layers']
        self.attn_factor = config['en.factor']
        self.dropout = config['en.dropout']
        # 定义网络结构
        self.conv = nn.Conv1d(in_channels=self.block_num, out_channels=self.out_channels, kernel_size=3, padding=1)
        # 全连接层需要cat h_t 和 c_t 以及输入的三个时间序列长度值 seq_len 导致需要在hidden_size * 2
        self.attn = nn.Sequential(
            nn.Linear(self.block_len, self.attn_factor * self.block_len),  # 这里只用到了一个全连接层，和论文中不一样
            nn.Tanh(),
            nn.Linear(self.attn_factor * self.block_len, 1)
        )
        self.lstm = nn.LSTM(
            input_size=self.out_channels,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=self.dropout
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        # input_data shape :batch,1,history   --->history 拆解为 block_num,block_len
        self.batch_size = input_data.shape[0]  # 当train 和test batch 不一致，需要重置self.batch_size
        # input_data 的初始维度为 (batch,seq_len,in_size) ---> (batch,block_num,block_len)
        input_data = input_data.view(-1, self.block_num, self.block_len)
        x = self.conv(input_data)  # shape  batch,out_channel,block_len
        # 适应rnn 的输入要求
        # 初始化h 使用高斯分布, (num_dir, batch, hidden_size)
        h_t = init_rnn_hidden(self.batch_size, self.hidden_size, num_layers=self.lstm_num_layers)
        c_t = init_rnn_hidden(self.batch_size, self.hidden_size, num_layers=self.lstm_num_layers)
        e_t = self.attn(x)
        alpha_t = self.softmax(e_t)  # batch,col,1
        x_hat = torch.mul(x, alpha_t)  # 进行加权
        x_hat = x_hat.permute(2, 0, 1)  # 适应rnn 输入 ->history==block_len,batch,out_channel
        x_out, (h_t, c_t) = self.lstm(x_hat, (h_t, c_t))
        return x_out, alpha_t  # 返回每个时刻输出的h_ti  -这里总共有 block_len 个


class SupEncoder(nn.Module):
    def __init__(self, config):
        super(SupEncoder, self).__init__()
        self.out_channel = config['en.sup_out_channel']
        self.conv = nn.Conv1d(in_channels=3, out_channels=self.out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape batch,col,history_diff 24 point
        out = self.conv(x)
        return out.permute(2, 0, 1)  # 适应 rnn history,batch,out_channel


class TemproalDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config['running.batch_size']
        self.code_size = config['en.hidden_size']   + config['en.sup_out_channel']
        self.block_num = config['common.block_num']
        self.block_len = config['common.block_len']
        self.hidden_size = config['de.hidden_size']
        self.lstm_num_layers = config['de.num_layers']
        self.prediction_horizon = config['common.prediction_horizon']
        self.dropout = config['de.dropout']

        self.fusion_con = ResFlusion(config)
        self.lstm = nn.LSTM(input_size=self.code_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_num_layers,
                            dropout=self.dropout
                            )
        # 这个版本将code拼接在fc out层进行调整
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size / 2), 1)
        )

    def forward(self, input_code: torch.Tensor, sup_code: torch.Tensor):
        self.batch_size = input_code.shape[1]
        # fusion_x = torch.cat([input_code, sup_code], dim=2)
        fusion_x = self.fusion_con(input_code, sup_code)
        h_t = init_rnn_hidden(self.batch_size, self.hidden_size, num_layers=self.lstm_num_layers)
        c_t = init_rnn_hidden(self.batch_size, self.hidden_size, num_layers=self.lstm_num_layers)
        x_hat, (h_t, c_t) = self.lstm(fusion_x, (h_t, c_t))
        out = self.fc_out(h_t[-1, :, :])
        return out


class ResFlusion(nn.Module):
    def __init__(self, config):
        super(ResFlusion, self).__init__()
        self.main_code_size = config['en.hidden_size']
        self.code_size = config['en.hidden_size'] + config['en.sup_out_channel']

        self.layer_norm = nn.LayerNorm([config['common.block_len'], self.code_size], elementwise_affine=False)
        self.res_con = nn.Sequential(
            nn.Linear(self.code_size, 2 * self.code_size),
            nn.Sigmoid(),
            nn.Linear(2 * self.code_size, self.code_size)
        )

    def forward(self, main_x, sup_x):
        mix = torch.cat([main_x, sup_x], dim=2)
        ln_mix = self.layer_norm(mix.permute(1, 0, 2))  # ---> batch,history,hidden_size
        # F.layer_norm(mix,sup_x)
        res_mix = self.res_con(ln_mix)
        fusion_x = res_mix + ln_mix
        return fusion_x.permute(1, 0, 2)  # --->history,batch,hidden_size

        # 不通过res_con
        # fusion_x = torch.cat([main_x, sup_x], dim=2)
        # ln_x = self.layer_norm(fusion_x.permute(1, 0, 2))
        # return fusion_x  # ln_x.permute(1, 0, 2)



    def configure_optimizers(self):
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        #optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        optimizer = torch.optim.AdamW([
            {'params': weight_p, 'weight_decay': 0.02},
            {'params': bias_p, 'weight_decay': 0}
        ], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96,verbose=True)
        #StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,100,150], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optim_dict
