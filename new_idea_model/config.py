import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
p2 = parents_config.copy()
p2["running.batch_size"] = 2
p2['common.history_seq_len'] = 168
p2["running.lr"]= 0.001
p2['common.prediction_horizon'] = 24 # 测试直接预测一个值即可
config = p2
config['test'] = False
# 日基线参数
config['day_baseline_input'] = 7
config['day_baseline_lstm_layer'] = 2
config['day_baseline_lstm_hidden'] = 64 # **调整**
# ar 参数
config['ar_hidden'] = 128 # 使用一周的数据 固定值
config['ar_out'] = 32 # **调整**
# 周基线参数
config['week_baseline_cnn_channel1'] = 16
config['week_baseline_cnn_out'] =  24 # 时间步长
# 预测层参数
config['predict_lstm_layer'] = 1
config['predict_lstm_hidden'] = 128 # **调整**
config['predict_fc_hidden'] = config['predict_lstm_hidden']
config['prediction_horizon'] = 1
#attn 参数
k_q = 64 # key和value 保持一样的长度
config['attn_value_in'] = config['week_baseline_cnn_out']
config['attn_value_out'] = k_q
config['attn_key_in'] = config['week_baseline_cnn_out']
config['attn_key_out'] = k_q
config['attn_query_in'] = config['predict_lstm_hidden']
config['attn_query_out'] = k_q
parameter = config
