import torch
from ray import tune
from common_config.common_config import parents_config

# ray.tune功能测试，只跑一个epoch
config = parents_config.copy()
config["running.batch_size"] = 4
config['common.history_seq_len'] = 168
config["running.lr"]= 0.001
config['test'] = False

# Full context 参数
config['full_baseline_lstm_hidden'] = 64
config['full_baseline_lstm_layer'] = 2
config['full_baseline_fc_hidden'] = 32
# time local 参数
config['time_baseline_input'] = 7  # 固定值 7
config['time_baseline_lstm_hidden'] = 64
config['time_baseline_lstm_layer'] = 1 # 取值1-3
config['time_baseline_adapter_out'] = 64
config['time_baseline_fc_hidden'] = config['time_baseline_adapter_out']

# date local 参数
config['date_baseline_input'] = 24  # 固定值24
config['date_baseline_lstm_hidden'] = 128
config['date_baseline_lstm_layer'] = 1 # 取值1-3
config['date_baseline_adapter_out'] = 64
config['date_baseline_fc_hidden'] = config['date_baseline_adapter_out']

# local CNN 参数
config['local_cnn_channel1'] = 16

#attn 参数

    # timelocal
time_k_q = 32 # key和query 保持一样的长度
config['time_attn_key_in'] = config['time_baseline_input'] #固定值
config['time_attn_key_out'] = time_k_q
config['time_attn_query_in'] = config['time_baseline_lstm_hidden']
config['time_attn_query_out'] = time_k_q
config['time_attn_value_in'] = config['time_attn_key_in']
config['time_attn_value_out'] = config['time_baseline_adapter_out']
    # datelocal
date_k_q = 32
config['date_attn_key_in'] = config['date_baseline_input'] # 固定值
config['date_attn_key_out'] = date_k_q
config['date_attn_query_in'] = config['date_baseline_lstm_hidden']
config['date_attn_query_out'] = date_k_q
config['date_attn_value_in'] = config['date_attn_key_in']
config['date_attn_value_out'] = config['time_baseline_adapter_out']


# 预测层参数
config['prediction_horizon'] = 1

parameter = config
