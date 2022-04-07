import os
import abc

import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl


class AbsDataModule(pl.LightningDataModule):

    def __init__(self,
                 target,
                 history_seq_len,
                 prediction_horizon=1,
                 batch_size=1,
                 srate=[0.7, 0.2, 0.1],
                 data_succession=True):
        '''
        在初始化中已经对数据集进行了切分
        :param history_seq_len: 使用的过去时间长度
        :param target:
        :param prediction_horizon:
        :param batch_size:
        :param data_succession:
        '''
        self.num_work = 1
        self.history_seq_len = history_seq_len
        self.batch_size = batch_size
        step = prediction_horizon if data_succession else history_seq_len
        data_size = target.shape[0] if type(target) == pd.DataFrame else len(target)
        x_seqs, y_seqs = [], []
        for i in range(0, data_size - history_seq_len - prediction_horizon, step):
            x = target[i:i + history_seq_len]
            y = target[i + history_seq_len:i + history_seq_len + prediction_horizon].tolist()
            x_seqs.append(torch.FloatTensor(x).view(1, -1)), y_seqs.append(torch.FloatTensor(y).view(1, -1))
        self.train_x, self.val_x, self.test_x = self._split(x_seqs, *srate)
        self.train_y, self.val_y, self.test_y = self._split(y_seqs, *srate)

    def _split(self, target, train_rate: float, val_rate: float, test_rate: float):
        assert round(train_rate + val_rate + test_rate, 2) == 1.0
        data_size = target.shape[0] if type(target) == pd.DataFrame else len(target)
        train_data = target[:int(data_size * train_rate)]
        val_data = target[int(data_size * train_rate):int(data_size * (train_rate + val_rate))]
        test_data = target[int(data_size * (train_rate + val_rate)):]
        return train_data, val_data, test_data

    def _to_dataset(self, x, y):
        x_var, y_var = torch.cat(x).view(-1, self.history_seq_len, 1), torch.cat(y)
        return TensorDataset(x_var, y_var)

    def setup(self, stage):
        if stage == 'fit':
            pass
        if stage == 'test':
            pass

    def train_dataloader(self):
        dataset = self._to_dataset(self.train_x, self.train_y)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def val_dataloader(self):
        dataset = self._to_dataset(self.val_x, self.val_y)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)

    def test_dataloader(self):
        dataset = self._to_dataset(self.train_x[:50], self.train_y[:50])
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)

    def predict_dataloader(self):
        dataset = self._to_dataset(self.test_x, self.test_y)
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)

    def get_test_data(self):
        return self.test_x, self.test_y
