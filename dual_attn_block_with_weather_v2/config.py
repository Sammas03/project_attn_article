import torch
from ray import tune

p1 = {
    # 编解码器中的公共参数

    "common.block_num": 6,  # 长周期使用的天数
    "common.block_len": 4,  # 长时序使用的序列长度
    # "common.short_time": 24,  # 短时序使用的序列长
    "common.prediction_horizon": 1,  # 预测的序列长度

    # 输入attention 编码器网络参数
    "input_encoder.hidden_size": 128,  # lstm 的input_size 跟输入的long_days 保持一致
    "input_encoder.directions": 1,  # rnn方向
    "input_encoder.output_size": 128,
    "input_encoder.attn_hidden": 128,
    "input_encoder.num_layers": 1,

    # 时序attention 和预测网络解码器参数
    "temporal_decoder.hidden_size": 128,
    "temporal_decoder.num_layers": 1,

    # 天气因素编码参数
    "weather.column_name": ['apparentTemperature', 'windSpeed', 'dewPoint'],
    "weather.out_size": 128,  # 根据 编码融合的位置不同进行调整
    "weather.hidden_size": 128,
    "weather.input_size": 3,
    "weather.num_layers": 2,

    # 运行时的一些参数
    "running.device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "running.lr": 1e-4,
    "running.batch_size": 8,
    "running.num_epoch": 1,
    "running.lrs_step_size": 2000,  # 学习率下降步长
    "running.max_grad_norm": 0.1,  # 梯度剪切 最大梯度
    "running.gradient_accumulation_steps": 1,  # 梯度累计计算步长
    "running.reg1": False,  # L1 正则化
    "running.reg2": True,  # L2 正则化
    "running.reg_factor1": 1e-4,
    "running.reg_factor2": 1e-4,
    "running.data_succession": True  # 样本数据是否连续

}

# ray.tune功能测试，只跑一个epoch
p2 = p1.copy()
p2['input_encoder.attn_hidden'] = tune.randint(64, 256)
p2['input_encoder.hidden_size'] = tune.randint(64, 256)
p2['group.hidden_size'] = tune.randint(32, 128)
p2['group.output_size'] = tune.randint(64, 256)
p2['temporal_decoder.hidden_size'] = tune.randint(64, 256)

# 服务器进行参数搜索使用
p3 = p2.copy()
p3["running.num_epoch"] = 600

parameter = p1
