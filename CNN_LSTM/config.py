import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2["running.lr"] = 0.0001
p2['input_size'] = 16  # 在cnn lstm 中是 cnn卷积之后的out_channel
p2['lstm.hidden_num'] = 128
p2['lstm.num_layers'] = 2
parameter = p2
