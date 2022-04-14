import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['input_size'] = 4
p2['lstm.hidden_num'] = tune.randint(128, 374)
p2['lstm.num_layers'] = 4
parameter = p2
