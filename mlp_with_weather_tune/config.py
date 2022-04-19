import torch
from ray import tune

from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['running.lr'] = 0.0001
p2['input_size'] = 4*parents_config['common.history_seq_len']
p2['mlp.layer1.hidden_num'] = 1024
p2['mlp.layer2.hidden_num'] = 288
p2['mlp.layer3.hidden_num'] = 215
p2['mlp.layer4.hidden_num'] = 79
parameter = p2
