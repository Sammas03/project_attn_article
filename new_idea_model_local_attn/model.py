import os
import math
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class MainModel(AbsModel):
    '''
        进行lstm神经网络预测的测试
    '''

    def __init__(self, config):
        super().__init__()
        self.predict_y = []
        self.real_y = []
        '''
        ####################pytorch设置###############################
        '''
        if (config['gpu']):
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)

        # 辅助参数
        self.lr = config['running.lr']
        self.seq_len = config['common.history_seq_len']

        '''
            #####################模型部分############################
        '''

        self.full_context = FullContextLayer(config)
        self.Local_cnn = LocalCNN(config)
        self.time_local_attn = TimeLocalAttn(config)
        self.date_local_attn = DateLocalAttn(config)
        # self.weight_init()

    def forward(self, x):
        # x dim(batch,seq_len,factor_num)
        # 准备临时变量
        batch = x.shape[0]
        device = torch.device('cuda:0')
        # ar 输出分量
        y_f = self.full_context(x)
        x_baseline = x.reshape(-1, 7, 24)  # 用作局部基线分析的x
        x_time = x_baseline.permute(0, 2, 1)  # 维度置换
        x_date = x_baseline
        v_cnn = self.Local_cnn(x_baseline)
        y_t = self.time_local_attn(x_time, v_cnn)
        y_d = self.date_local_attn(x_date, v_cnn)
        y_hat = y_f + y_t + y_d
        return y_hat

    def configure_optimizers(self):
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.AdamW([
        #     {'params': weight_p, 'weight_decay': 0.1},
        #     {'params': bias_p, 'weight_decay': 0}
        # ], lr=self.lr)
        StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.88, verbose=True)
        # StepLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 35, 50, 75, 100], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict


class FullContextLayer(nn.Module):
    def __init__(self, config):
        super(FullContextLayer, self).__init__()
        '''
        ###################参数部分##############################
        '''
        self.full_baseline_lstm_input = 1
        self.full_baseline_lstm_hidden = config['full_baseline_lstm_hidden']
        self.full_baseline_lstm_layer = config['full_baseline_lstm_layer']
        self.full_baseline_fc_hidden = config['full_baseline_fc_hidden']
        '''
            #####################模型部分############################
        '''
        self.ar = nn.LSTM(input_size=self.full_baseline_lstm_input,
                          hidden_size=self.full_baseline_lstm_hidden,
                          num_layers=self.full_baseline_lstm_layer,
                          dropout=0.5
                          )
        self.fc_out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.full_baseline_lstm_hidden, int(self.full_baseline_fc_hidden / 2)),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(int(self.full_baseline_fc_hidden / 2), 1)
        )

    def forward(self, x):
        batch = x.shape[0]
        x = x.permute(1, 0, 2)  # 适应rnn
        (h_0, c_0) = (
            init_rnn_hidden(batch=batch,
                            hidden_size=self.full_baseline_lstm_hidden,
                            num_layers=self.full_baseline_lstm_layer),
            init_rnn_hidden(batch=batch,
                            hidden_size=self.full_baseline_lstm_hidden,
                            num_layers=self.full_baseline_lstm_layer)
        )
        h, _ = self.ar(x, (h_0, c_0))
        y_ar = self.fc_out(h[-1, :, :])
        return y_ar


class TimeLocalAttn(nn.Module):
    def __init__(self, config):
        super(TimeLocalAttn, self).__init__()
        '''
            ###################参数部分##############################
        '''
        self.time_baseline_input = config['time_baseline_input']  # 固定值 7
        self.time_baseline_lstm_hidden = config['time_baseline_lstm_hidden']
        self.time_baseline_lstm_layer = config['time_baseline_lstm_layer']
        self.time_baseline_adapter_out = config['time_baseline_adapter_out']
        self.time_baseline_fc_hidden = config['time_baseline_fc_hidden']
        '''
            #####################模型部分############################
        '''
        self.time_baseline_lstm = nn.LSTM(input_size=self.time_baseline_input,
                                          hidden_size=self.time_baseline_lstm_hidden,
                                          num_layers=self.time_baseline_lstm_layer,
                                          dropout=0.5
                                          )
        self.time_baseline_adapter = nn.Linear(self.time_baseline_lstm_hidden,
                                               self.time_baseline_adapter_out)
        self.attn = Attn(config,'time')

        self.fc_out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.time_baseline_fc_hidden, int(self.time_baseline_fc_hidden / 2)),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(int(self.time_baseline_fc_hidden / 2), 1)
        )

    def forward(self, x, v_cnn):
        batch = x.shape[0]
        x = x.permute(1, 0, 2)
        (h_0, c_0) = (
            init_rnn_hidden(batch=batch,
                            hidden_size=self.time_baseline_lstm_hidden,
                            num_layers=self.time_baseline_lstm_layer),
            init_rnn_hidden(batch=batch,
                            hidden_size=self.time_baseline_lstm_hidden,
                            num_layers=self.time_baseline_lstm_layer),
        )
        h, _ = self.time_baseline_lstm(x, (h_0, c_0))
        v_hat = self.attn(v_cnn.permute(0,2,1), h[-1, :, :])
        x_star = torch.mul(self.time_baseline_adapter(h[-1,:,:]), v_hat)
        y_t = self.fc_out(x_star)
        return y_t


class DateLocalAttn(nn.Module):
    def __init__(self, config):
        super(DateLocalAttn, self).__init__()
        '''
            ###################参数部分##############################
        '''
        self.date_baseline_input = config['date_baseline_input']  # 固定值 7
        self.date_baseline_lstm_hidden = config['date_baseline_lstm_hidden']
        self.date_baseline_lstm_layer = config['date_baseline_lstm_layer']
        self.date_baseline_adapter_out = config['date_baseline_adapter_out']
        self.date_baseline_fc_hidden = config['date_baseline_fc_hidden']
        '''
            ###################参数部分##############################
        '''
        self.time_baseline_lstm = nn.LSTM(input_size=self.date_baseline_input,
                                          hidden_size=self.date_baseline_lstm_hidden,
                                          num_layers=self.date_baseline_lstm_layer,
                                          dropout=0.5
                                          )
        self.date_baseline_adapter = nn.Linear(self.date_baseline_lstm_hidden,
                                               self.date_baseline_adapter_out)

        self.attn = Attn(config,'date')

        self.fc_out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.date_baseline_fc_hidden, int(self.date_baseline_fc_hidden / 2)),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(int(self.date_baseline_fc_hidden / 2), 1)
        )

    def forward(self, x, v_cnn):
        batch = x.shape[0]
        x = x.permute(1, 0, 2)
        (h_0, c_0) = (
            init_rnn_hidden(batch=batch,
                            hidden_size=self.date_baseline_lstm_hidden,
                            num_layers=self.date_baseline_lstm_layer),
            init_rnn_hidden(batch=batch,
                            hidden_size=self.date_baseline_lstm_hidden,
                            num_layers=self.date_baseline_lstm_layer),
        )
        h, _ = self.time_baseline_lstm(x, (h_0, c_0))
        v_hat = self.attn(v_cnn, h[-1, :, :])
        x_star = torch.mul(self.date_baseline_adapter(h[-1,:,:]), v_hat)
        y_d = self.fc_out(x_star)
        return y_d


class LocalCNN(nn.Module):
    def __init__(self, config):
        super(LocalCNN, self).__init__()
        self.local_cnn_channel1 = config['local_cnn_channel1']
        self.local_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.local_cnn_channel1,
                      padding=1, kernel_size=(3, 3)),
            nn.Tanh(),
            nn.Conv2d(in_channels=self.local_cnn_channel1, out_channels=1,
                      padding=1, kernel_size=(3, 3))
        )

    def forward(self, x):
        # x dim(batch,seq_len,)
        x_local = x.unsqueeze(1)
        y_c = self.local_cnn(x_local)
        return y_c.squeeze(1)  # 通道数在cnn分析完成后去除


class Attn(nn.Module):

    def __init__(self, config, perfix):
        super().__init__()
        self.config = config
        # attn 参数
        self.value_in = config[perfix + '_attn_value_in']
        self.value_out = config[perfix + '_attn_value_out']
        self.key_in = config[perfix + '_attn_key_in']
        self.key_out = config[perfix + '_attn_key_out']
        self.query_in = config[perfix + '_attn_query_in']
        self.query_out = config[perfix + '_attn_query_out']
        self.value = nn.Linear(self.value_in, self.value_out, bias=False)
        self.key = nn.Linear(self.key_in, self.key_out, bias=False)
        # 对LSTM的隐藏状态h-->转换为查询向量query
        self.query = nn.Linear(self.query_in, self.query_out, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h):
        sqrt_d_k = math.sqrt(self.key_out)
        factor_num = x.shape[-1]  # 待attn时间序列的因素数，dim（batch,seq_len,factor_num）
        # h维度(batch, hidden_size)
        # 1. h是张量形式 需要复制成矩阵形式，然后进行转置, 并映射到query
        # h.repeat(factor_num,1,1) 不需要
        h1 = self.query(h)
        query = h1.unsqueeze(2)  # (batch,query_size,1)
        # 2. 对一组X 独立进行两次线性变换 得到 key,value
        key = self.key(x)  # dim(batch,seq_len,key_size)
        value = self.value(x)  # dim(batch seq_len,value_size)
        # 3. 通过矩阵点积计算得到权重向量a
        a = torch.bmm(key, query) / sqrt_d_k  # batch不参与运算,(seq_len,key_size) * (query_size,1) = (seq_len,1)
        # 4. 计算 a*softmax(v) 得到最终的加权向量 x_hat
        a_hat = self.softmax(a)
        #             (1,seq_len) *(seq_len,value_size) = (1,value_size)
        x_hat = torch.bmm(a_hat.permute(0, 2, 1), value).squeeze(1)  # dim(batch,value_size)
        return x_hat  # 适配rnn dim(seq_len,batch,factor_num),factor_num == value_out
