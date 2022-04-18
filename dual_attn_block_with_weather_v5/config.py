import torch
from ray import tune
from common_config.common_config import parents_config

p2 = parents_config.copy()
p2["running.lr"] = 0.001
p2['en.hidden_size'] = 32
p2['en.out_channels'] = 16
p2['en.num_layers'] = 2
p2['en.factor'] = 3  # attn linear hidden size
p2['en.sup_out_channel'] = 16  # 每个时刻的sup 因素编码大小
p2['de.hidden_size'] = 64
p2['de.num_layers'] = 2
p2['en.dropout'] = 0.6
p2['de.dropout'] = 0.5
p2['ar.layer1'] = 72 * 4
p2['ar.dropout'] = 0.5
# 服务器进行参数搜索使用
parameter = p2
# 'r2': 0.41365325208541237

# 'r2': 0.4318737365808861 de hidden_size =64

# 'r2': 0.4489776635442433 开启指数lr衰减 gamma =0.9

# {'r2': 0.48196688448490876, 'evs': 0.48303332312149105, 'mae': 0.04217949445452987, 'mse': 0.0032897147865119276, 'rmse': 0.057356035310261184}

'''
p2["running.lr"]= 0.001
p2['en.hidden_size'] = 32
p2['en.out_channels'] = 16
p2['en.num_layers'] = 2
p2['en.factor'] = 3  # attn linear hidden size
p2['en.sup_out_channel'] = 16 # 每个时刻的sup 因素编码大小
p2['de.hidden_size'] = 64
p2['de.num_layers'] = 2
p2['en.dropout'] = 0.5
p2['de.dropout'] = 0.5
p2['ar.layer1'] = 72 * 4
p2['ar.dropout'] = 0.5


'''
