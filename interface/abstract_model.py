import os
import abc
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from util.metric_util import easy_metric
from util.plot_util import easy_plot
from pytorch_lightning import seed_everything
seed_everything(2022)

class AbsModel(pl.LightningModule):
    '''
        进行gru神经网络预测的测试
    '''

    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.mse_loss(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # if self.predict_y is None : self.predict_y = []
        # if self.real_y is None : self.real_y = []
        # self.predict_y.append(y_hat)
        # self.real_y.append(y)
        return {'real_y': y.cpu().numpy().tolist(), 'predict_y': y_hat.cpu().numpy().tolist()}

    def predict_step(self, batch, batch_idx: int):
        x, y = batch
        y_hat = self(x)
        return {'real_y': y.cpu().numpy().tolist(), 'predict_y': y_hat.cpu().numpy().tolist()}

    def test_epoch_end(self, outputs):
        '''
        在测试完成后进行汇总和画图
        :param outputs:
        :return:
        '''
        reals, predicts = [], []
        for ite in outputs:
            reals.extend(ite['real_y']), predicts.extend(ite['predict_y'])
        easy_plot(reals=reals, predicts=predicts, title="train result")

    def on_predict_epoch_end(self, outputs):
        reals, predicts = [], []
        for ite in outputs[0]:
            reals.extend(ite['real_y']), predicts.extend(ite['predict_y'])
        _, (evs, mae, mse, r2) = easy_metric(reals, predicts)
        print((evs, mae, mse, r2))
        easy_plot(reals=reals, predicts=predicts, title="predict result")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.5)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict

    def mse_loss(self, x, y):
        loss = nn.MSELoss()(x, y)

        # params = torch.cat([p.view(-1) for name, p in self.named_parameters() if 'bias' not in name])
        # loss += 1e-3 * torch.norm(params, 2)
        return loss
