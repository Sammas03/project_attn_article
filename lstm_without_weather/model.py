import os
import torch
from torch import nn
import pytorch_lightning as pl
from util.nn_util import init_rnn_hidden
from interface.abstract_model import AbsModel


class LstmModel(AbsModel):
    '''
        进行lstm神经网络预测的测试
    '''

    def __init__(self, config):
        super().__init__()
        '''
        ####################pytorch设置###############################
        '''
        if (config['gpu']):
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)
        self.predict_y = []
        self.real_y = []
        hidden_num = config['lstm.hidden_num']
        output_num = config['output_size']
        self.hidden_size = hidden_num
        self.lr = config['running.lr']
        self.input_size = config['input_size']
        self.layers = config['lstm.num_layers']
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=self.hidden_size,
                            num_layers=self.layers,
                            dropout=0.1
                            )
        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(self.hidden_size, output_num)
        )
        self.weight_init()

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # x = x.permute(2, 0, 1)  # seq_len,batch,input_dim
        x = x.permute(1,0,2)
        seq_len, batch, input_dim = x.shape
        # h = init_rnn_hidden(batch=batch, hidden_size=self.hidden_size, num_layers=self.layers)
        # c = init_rnn_hidden(batch=batch, hidden_size=self.hidden_size, num_layers=self.layers)
        y, (h, c) = self.lstm(x)
        out = self.fc_out(h[-1, :, :])
        return out

    #
    #
    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.mse_loss(y_hat, y)
    #     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss
    #
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.mse_loss(y_hat, y)
    #     self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = self.mse_loss(y_hat, y)
    #     self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     # if self.predict_y is None : self.predict_y = []
    #     # if self.real_y is None : self.real_y = []
    #     # self.predict_y.append(y_hat)
    #     # self.real_y.append(y)
    #     return {'real_y': y.cpu().numpy().tolist(), 'predict_y': y_hat.cpu().numpy().tolist()}
    #
    #
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
    #     optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
    #     return optim_dict
    #
    # def mse_loss(self, x, y):
    #     loss = nn.MSELoss()(x, y)
    #     # params = torch.cat([p.view(-1) for name, p in self.named_parameters() if 'bias' not in name])
    #     # loss += 1e-3 * torch.norm(params, 2)
    #     return loss


    def configure_optimizers(self):
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        # optimizer = torch.optim.AdamW([
        #     {'params': weight_p, 'weight_decay': 0.05},
        #     {'params': bias_p, 'weight_decay': 0}
        # ], lr=self.lr)
        StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98,verbose=True)
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,75,100], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

