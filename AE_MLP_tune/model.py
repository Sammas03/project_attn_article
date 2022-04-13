import os
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class AeMlpModel(AbsModel):
    '''
        进行gru神经网络预测的测试
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = config['running.lr']  # configure_optimizers使用
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(x)
        # cat_code = torch.cat(encoded, dim=1)
        out = self.decoder(encoded)
        return out


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
        super(Encoder, self).__init__()
        self.config = config
        self.layer1 = Mlp_Uint(config)
        self.layer2 = Mlp_Uint(config)
        self.layer3 = Mlp_Uint(config)
        self.layer4 = Mlp_Uint(config)

    def forward(self, x):
        out1 = self.layer1(x[:, 0, :])
        out2 = self.layer2(x[:, 1, :])
        out3 = self.layer3(x[:, 2, :])
        out4 = self.layer4(x[:, 3, :])
        return torch.cat((out1, out2, out3, out4), dim=1)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.decoder = Mlp_Uint(config,
                                input_size=config['aemlp.encode_size'] * 4,
                                layer1=config['de.layer1'],
                                layer2=config['de.layer2'],
                                layer3=config['de.layer3'],
                                out_size=config['de.layer3'])
        self.fc_out = nn.Sequential(  # 最后一层采用relu函数剔除掉负值
            nn.Tanh(),
            nn.Linear(config['de.layer3'], config['output_size'])
        )

    def forward(self, x):
        out = self.fc_out(self.decoder(x))
        return out
