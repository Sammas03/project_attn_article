import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2['running.lr'] = 0.0001
p2['unit.layer1.hidden_num'] = 1024
p2['unit.layer2.hidden_num'] = 576
p2['unit.layer3.hidden_num'] = 432
p2['aemlp.encode_size'] = 420
p2['de.layer1'] = 1024
p2['de.layer2'] = 256
p2['de.layer3'] = 128
# p2['running.min_epoch'] = 150
parameter = p2
