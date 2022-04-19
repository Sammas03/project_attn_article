import os
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class MlpModel(AbsModel):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = config['running.lr']  # configure_optimizers使用
        self.predict_y = []
        self.real_y = []
        input_size = config['input_size']
        hidden_num_1 = config['mlp.layer1.hidden_num']
        hidden_num_2 = config['mlp.layer2.hidden_num']
        hidden_num_3 = config['mlp.layer3.hidden_num']
        hidden_num_4 = config['mlp.layer4.hidden_num']
        output_num = config['output_size']
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_num_1),
            nn.Tanh(),
            nn.Linear(hidden_num_1, hidden_num_2),
            nn.Tanh(),
            nn.Linear(hidden_num_2, hidden_num_3),
            nn.Tanh(),
            nn.Linear(hidden_num_3, hidden_num_4),
            nn.Tanh(),
            nn.Linear(hidden_num_4, output_num)
        )

        self.weight_init()

    def forward(self, x: torch.Tensor):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # x原始维度，batch,seq_len,input_dim ，input_dim==1
        batch, seq_len, input_dim = x.shape
        x = x.view(batch, seq_len * input_dim)
        out = self.mlp(x)
        return out
