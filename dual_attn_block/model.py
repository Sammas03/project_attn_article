import os
import torch
from torch.autograd import Variable
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
        self.predict_y = []
        self.real_y = []
        hidden_num = config['gru.hidden_num']
        output_num = config['gru.output_num']
        num_layers = config['gru.num_layers']
        self.hidden_size = hidden_num
        self.lr = config['running.lr']
        self.encoder = InputEncoder(config)
        self.decoder = TemproalDecoder(config)
        self.hidden = None

    def forward(self, x):
        encoded, attn = self.encoder(x)
        output = self.decoder(encoded, x)
        return output


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def mse_loss(self, x, y):
        # loss = nn.MSELoss()(x, y)
        # params = torch.cat([p.view(-1) for name, p in self.named_parameters() if 'bias' not in name])
        # loss += 1e-3 * torch.norm(params, 2)
        return nn.MSELoss()(x, y)


class InputEncoder(nn.Module):

    def __init__(self, config):
        super(InputEncoder, self).__init__()
        self.batch_size = config['running.batch_size']
        self.block_num = config['common.block_num']
        self.block_len = config['common.block_len']
        self.hidden_size = config['input_encoder.hidden_size']
        # 全连接层需要cat h_t 和 c_t 以及输入的三个时间序列长度值 seq_len 导致需要在hidden_size * 2
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_size + self.block_len, self.hidden_size + self.block_len),  # 这里只用到了一个全连接层，和论文中不一样
            nn.Tanh(),
            nn.Linear(self.hidden_size + self.block_len, 1)
        )
        self.gru = nn.GRU(
            input_size=self.block_num,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        self.batch_size = input_data.shape[0] #当train 和test batch 不一致，需要重置self.batch_size
        # input_data 的初始维度为 (batch,seq_len,in_size) ---> (batch,block_num,block_len)
        input_data = input_data.view(-1, self.block_num, self.block_len)
        # 这里为了attention计算需要将tensor 转换到 （batch,block_len,block_num）
        input_data = input_data.permute(0, 2, 1)
        # 初始化h 使用高斯分布, (num_dir, batch, hidden_size)
        h_t = init_rnn_hidden(self.batch_size, self.hidden_size, num_layers=1)

        # 初始化 attention 和 编码 跟input的数据抱持一致
        attentions, input_encoded = (Variable(torch.zeros(self.batch_size, self.block_len, self.block_num)),
                                     Variable(torch.zeros(self.batch_size, self.block_len, self.hidden_size)))

        for t in range(self.block_len):
            # repeat 会将tensor 在指定的维度上重复
            # 对于输入到神经网络的factor 都需要一个 h_t
            # x 的维度：batch,block_num,hidden_size + block_len
            x = torch.cat((h_t.repeat(self.block_num, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)
            e_t = self.attn(x)  # liner输入 batch,block_num,hidden_size+block_len。每个batch执行block_num次全连接计算
            a_t = self.softmax(e_t)  # e_t (batch,block_num,1)
            # mul是矩阵对应位相乘， 将输入每天的的t时刻数据取出与对应的attention相乘构成 x_t
            x_t = torch.mul(a_t, input_data[:, t, :].unsqueeze(2))  # 乘以最后一维，batch,block_num,1
            self.gru.flatten_parameters()
            _, h_t = self.gru(x_t.permute(2, 0, 1), h_t)  # 计算编码，这里为了简便，将最后一维置换到第一维度seq_len == 1
            input_encoded[:, t, :] = h_t  # 对应时间点的编码
            attentions[:, t, :] = a_t.squeeze(2)

        return input_encoded, attentions


class TemproalDecoder(nn.Module):
    def __init__(self, config):
        super(TemproalDecoder, self).__init__()
        self.batch_size = config['running.batch_size']
        self.code_size = config['input_encoder.hidden_size']
        self.block_num = config['common.block_num']
        self.block_len = config['common.block_len']
        self.decoder_hidden_size = config['temporal_decoder.hidden_size']
        self.prediction_horizon = config['common.prediction_horizon']


        self.attn = nn.Sequential(
            nn.Linear(self.decoder_hidden_size + self.code_size, self.decoder_hidden_size + self.code_size),
            nn.Tanh(),
            nn.Linear(self.decoder_hidden_size + self.code_size, 1)
        )

        self.fc = nn.Linear(self.code_size + 1, self.code_size + 1)
        self.decoder = nn.GRU(input_size=self.code_size + 1,
                              hidden_size=self.decoder_hidden_size,
                              num_layers=1)
        self.fc_out = nn.Linear(self.decoder_hidden_size, 1)
        self.fc.weight.data.normal_()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_encoded: torch.Tensor, his: torch.Tensor = None):
        self.batch_size = his.shape[0]
        temp_p = his[:, -1, :]  # x 的初始维度为 (batch,seq_len,in_size)
        output = torch.zeros(self.batch_size, self.prediction_horizon,1, requires_grad=False)
        # input_encoded(batch,block_len,code_size)
        # h_t, c_t = (
        #     init_hidden(input_encoded, self.decoder_hidden_size),
        #     init_hidden(input_encoded, self.decoder_hidden_size)
        # )
        d_t = init_rnn_hidden(self.batch_size, self.decoder_hidden_size, num_layers=1)

        # 0时刻初始化c
        attn = torch.zeros(self.batch_size, self.prediction_horizon, self.block_len)

        for t in range(self.prediction_horizon):  # 这里是 seq2seq 循环控制需要预测多少个值
            x = torch.cat((d_t.repeat(self.block_len, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # context = Variable(torch.zeros(input_encoded.size(0), self.code_size))
            # linear 也可以处理三维 tensor
            l_t = self.attn(x)
            b_t = self.softmax(l_t)
            # 这里使用批量tensor 乘法,需要batch相等，后两维使用矩乘法 （b,p,m） * （b,m,q） = （b,p,q）
            c_t = torch.bmm(b_t.permute(0, 2, 1), input_encoded)  # (batch_size, code_size)
            # TODO  fc 连接层的意义需要重新看论文
            y_tilde = self.fc(torch.cat((c_t, temp_p.unsqueeze(2)), dim=2))  # (batch_size, out_size)
            self.decoder.flatten_parameters()
            _, d_t = self.decoder(y_tilde.permute(1,0,2), d_t)
            out = self.fc_out(d_t.permute(1,0,2))  # predicting value at t=self.seq_length+1
            temp_p = out.view(self.batch_size,-1)
            output[:, t,:] = out.view(self.batch_size,-1)
            attn[:, t, :] = b_t.view(self.batch_size, -1)

        return output.squeeze(2) # 最后一维默认为1，直接压缩掉
