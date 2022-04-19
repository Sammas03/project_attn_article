import torch
from ray import tune
from common_config.common_config import parents_config


# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2["running.lr"]= 0.0001
p2['input_size']=4
p2['en.hidden_num'] = 64
p2['en.num_layers'] = 4
p2['de.hidden_num'] = 256
parameter = p2
