import torch
from torch import nn
import numpy as np
# k
# ^
# k_hat =k+(k−1)(d−1)
# padding = (k-1)/2
# 默认状态下dilation = 1

'''
nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, dilation=2, padding=1)
nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, padding=2)
nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, dilation=2, padding=3)
nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=2, padding=4)
nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=3, padding=3)
nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, dilation=3, padding=6)

'''

a = torch.tensor(np.ones(72), dtype=torch.float32)
a = a.reshape(1, 1, 72)
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=3, padding=3)
b = conv(a)
print(b.shape)
