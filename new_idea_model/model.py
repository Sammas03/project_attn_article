import os
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
        '''
        ###################参数部分##############################
        '''
        # 辅助参数
        self.lr = config['running.lr']
        self.seq_len = config['common.history_seq_len']
        # 日基线参数
        self.day_baseline_input = config['day_baseline_input']
        self.day_baseline_lstm_layer = config['day_baseline_lstm_layer']
        self.day_baseline_lstm_hidden = config['day_baseline_lstm_hidden']
        # ar 参数
        self.ar_hidden = config['ar_hidden']
        self.ar_out = config['ar_out']
        # 周基线参数
        self.week_baseline_cnn_channel1 = config['week_baseline_cnn_channel1']
        self.week_baseline_out = config['week_baseline_cnn_out']
        # 预测层参数
        self.predict_lstm_hidden = config['predict_lstm_hidden']
        self.predict_lstm_layer = config['predict_lstm_layer']
        self.predict_fc_hidden = config['predict_fc_hidden']
        self.prediction_horizon = config['prediction_horizon']
        # self.prediction_lstm_in = self.ar_out + self.day_baseline_lstm_hidden + config['attn_value_out']
        self.prediction_lstm_in = self.day_baseline_lstm_hidden + config['attn_value_out']

        '''
            #####################模型部分############################
        '''
        # 日基线提取模块
        self.day_baseline_lstm = nn.LSTM(input_size=self.day_baseline_input,
                                         hidden_size=self.day_baseline_lstm_hidden,
                                         num_layers=self.day_baseline_lstm_layer,
                                         dropout=0.1
                                         )
        # ar 自回归模块
        self.ar = nn.Sequential(
            nn.Linear(self.seq_len, self.ar_hidden),
            nn.Tanh(),
            nn.Linear(self.ar_hidden, self.ar_out)
        )
        # 周基线提取模块 padding=1代表与day_baseline维度保持一致,卷积没有使特征图变小
        self.week_baseline_cnn1 = nn.Conv2d(in_channels=1, out_channels=self.week_baseline_cnn_channel1,
                                            padding=1, kernel_size=(3, 3))
        self.week_baseline_cnn2 = nn.Conv2d(in_channels=self.week_baseline_cnn_channel1, out_channels=1,
                                            padding=1, kernel_size=(3, 3))
        self.week_baseline_cnn = nn.Sequential(
            self.week_baseline_cnn1,
            nn.Tanh(),
            self.week_baseline_cnn2
        )
        # 与day_baseline保持一致
        self.week_baseline_fc = nn.Linear(self.day_baseline_input, self.week_baseline_out)
        # 解码LSTM模块
        self.lstm2 = nn.LSTM(input_size=self.prediction_lstm_in,
                             hidden_size=self.predict_lstm_hidden,
                             num_layers=self.predict_lstm_layer,
                             dropout=0.5)
        # attn模块
        self.attn = Attn(config)
        # 输出模块
        self.fc_out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.predict_fc_hidden, int(self.predict_fc_hidden / 2)),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(int(self.predict_fc_hidden / 2), 1)
        )


        self.weight_init()

    def forward(self, x):
        # 准备临时变量
        batch_size = x.shape[0]
        device = torch.device('cuda:0')
        y_hat = torch.zeros(batch_size, self.prediction_horizon, 1,
                            requires_grad=False, device=device)
        # ar 输出分量
        y_ar = self.ar(x.permute(0, 2, 1))
        x_baseline = x.reshape(-1, 7, 24)  # 用作基线分析的x
        x_baseline = x_baseline.permute(2, 0, 1)  # 维度置换
        # day baseline 输出分量
        y_day, _ = self.day_baseline_lstm(x_baseline)
        y_day = y_day[-1, :, :]
        # week baseline 输出分量
        x_week_baseline = x_baseline.unsqueeze(0).permute(2, 0, 1, 3)
        y_week_t0 = self.week_baseline_cnn(x_week_baseline)
        y_week = y_week_t0.squeeze(1)  # 通道数在cnn分析完成后去除
        # 预测层
        (h_week, c_week) = (
            init_rnn_hidden(batch=batch_size,
                            hidden_size=self.predict_lstm_hidden,
                            num_layers=self.predict_lstm_layer),
            init_rnn_hidden(batch=batch_size,
                            hidden_size=self.predict_lstm_hidden,
                            num_layers=self.predict_lstm_layer)
        )
        for t in range(self.prediction_horizon):
            v_hat = self.attn(y_week.permute(0, 2, 1), h_week[-1, :, :])  #
            # x_cat = torch.cat((y_ar, y_day.unsqueeze(1), v_hat), dim=2)  # 维度统一为dim(batch,seq_len,factory_num)
            x_cat = torch.cat((y_day.unsqueeze(1), v_hat), dim=2)  # 维度统一为dim(batch,seq_len,factory_num)

            x_t, (h_week, c_week) = self.lstm2(x_cat.permute(1, 0, 2), (h_week, c_week))
            y_t = self.fc_out(x_t.squeeze(0))  # 第0 维度恒为1，线性输出中是不需要的
            y_hat[:, t, :] = y_t
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # x = x.permute(2, 0, 1)  # seq_len,batch,input_dim
        return y_hat.reshape(batch_size, -1)

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
        StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96,verbose=True)
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 35, 50, 75, 100], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict


class Attn(nn.Module):

    def __init__(self, config):
        super().__init__()
        # attn 参数
        value_in = config['attn_value_in']
        value_out = config['attn_value_out']
        key_in = config['attn_key_in']
        key_out = config['attn_key_out']
        query_in = config['attn_query_in']
        query_out = config['attn_query_out']
        self.value = nn.Linear(value_in, value_out)
        self.key = nn.Linear(key_in, key_out)
        # 对LSTM的隐藏状态h-->转换为查询向量query
        self.query = nn.Linear(query_in, query_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h):
        factor_num = x.shape[-1]  # 待attn时间序列的因素数，dim（batch,seq_len,factor_num）
        # h维度(num_layers * num_directions = 1, batch, hidden_size)
        # 1. h是张量形式 需要复制成矩阵形式，然后进行转置, 并映射到query
        # h.repeat(factor_num,1,1) 不需要
        h1 = h.unsqueeze(1)  # (batch,factor_num,hidden_size = seq_len)
        query = self.query(h1)
        # 2. 对一组X 独立进行两次线性变换 得到 key,value
        # x1 = x.permute(0, 2, 1)  # 这里可能要做一个变换适应linear计算的维度 dim(batch,factor_num,seq_len)
        key = self.key(x)
        value = self.value(x)
        # 3. 通过矩阵点积计算得到权重向量a
        a = torch.bmm(key, query.permute(0, 2, 1))
        # 4. 计算 a*softmax(v) 得到最终的加权向量 x_hat
        a_hat = self.softmax(a)
        x_hat = torch.bmm(a_hat.permute(0, 2, 1), value)
        return x_hat  # 适配rnn dim(seq_len,batch,factor_num),factor_num == value_out
