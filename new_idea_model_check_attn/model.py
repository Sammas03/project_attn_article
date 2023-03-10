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
        self.day_baseline_fc_out = config['day_baseline_fc_out']
        # ar 参数
        self.ar_hidden = config['ar_hidden']
        self.ar_out = config['ar_out']
        # 周基线参数
        self.week_baseline_cnn_channel1 = config['week_baseline_cnn_channel1']
        self.week_baseline_out = config['week_baseline_cnn_out']
        # 预测层参数
        self.predict_fc_hidden = self.day_baseline_fc_out
        self.prediction_horizon = config['prediction_horizon']

        '''
            #####################模型部分############################
        '''
        # 日基线提取模块
        self.day_baseline_lstm = nn.LSTM(input_size=self.day_baseline_input,
                                         hidden_size=self.day_baseline_lstm_hidden,
                                         num_layers=self.day_baseline_lstm_layer,
                                         dropout=0.5
                                         )
        # 与day_baseline保持一致
        self.day_baseline_fc = nn.Linear(self.day_baseline_lstm_hidden,
                                         self.day_baseline_fc_out,
                                         bias=False)

        # ar 自回归模块
        self.ar = nn.Sequential(
            nn.Linear(self.seq_len, self.ar_hidden),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(self.ar_hidden, self.ar_out)
        )

        # 周基线提取
        # 周基线提取模块 padding=1代表与day_baseline维度保持一致,卷积没有使特征图变小
        self.week_baseline_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.week_baseline_cnn_channel1,
                      padding=2, kernel_size=(5, 5)),
            nn.Tanh(),
            nn.Conv2d(in_channels=self.week_baseline_cnn_channel1, out_channels=1,
                      padding=2, kernel_size=(5, 5))
        )
        # attn模块
        self.attn = Attn(config)
        # 输出模块
        self.fc_out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.predict_fc_hidden, int(self.predict_fc_hidden / 2)),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(int(self.predict_fc_hidden / 2), 1)
        )
        # self.weight_init()

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

        # 行为自注意力
        v_hat = self.attn(y_week.permute(0, 2, 1), y_day)
        # x_attn = self.day_baseline_fc(y_day) + v_hat # 这里将注意力叠加到 y_day，也可以使用bmm or cat，并可以考虑残差网络
        x_attn = torch.mul(self.day_baseline_fc(y_day),v_hat)
        # 预测
        y_day2 = self.fc_out(x_attn)
        y_hat = y_day2 + y_ar.squeeze(1)
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
        StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.88,verbose=True)
        # StepLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 35, 50, 75, 100], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict


class Attn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # attn 参数
        value_in = config['attn_value_in']
        value_out = config['attn_value_out']
        key_in = config['attn_key_in']
        key_out = config['attn_key_out']
        query_in = config['attn_query_in']
        query_out = config['attn_query_out']
        self.value = nn.Linear(value_in, value_out,bias=False)
        self.key = nn.Linear(key_in, key_out,bias=False)
        # 对LSTM的隐藏状态h-->转换为查询向量query
        self.query = nn.Linear(query_in, query_out,bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h):

        sqrt_d_k = math.sqrt(self.config['attn_key_out'])
        factor_num = x.shape[-1]  # 待attn时间序列的因素数，dim（batch,seq_len,factor_num）
        # h维度(batch, hidden_size)
        # 1. h是张量形式 需要复制成矩阵形式，然后进行转置, 并映射到query
        # h.repeat(factor_num,1,1) 不需要
        h1 = self.query(h)
        query = h1.unsqueeze(2)  # (batch,query_size,1)
        # 2. 对一组X 独立进行两次线性变换 得到 key,value
        key = self.key(x) # dim(batch,seq_len,key_size)
        value = self.value(x) # dim(batch seq_len,value_size)
        # 3. 通过矩阵点积计算得到权重向量a
        a = torch.bmm(key, query) /sqrt_d_k # batch不参与运算,(seq_len,key_size) * (query_size,1) = (seq_len,1)
        # 4. 计算 a*softmax(v) 得到最终的加权向量 x_hat
        a_hat = self.softmax(a)
        #             (1,seq_len) *(seq_len,value_size) = (1,value_size)
        x_hat = torch.bmm(a_hat.permute(0, 2, 1), value).squeeze(1) # dim(batch,value_size)
        return x_hat  # 适配rnn dim(seq_len,batch,factor_num),factor_num == value_out
