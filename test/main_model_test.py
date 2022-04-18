from ray import tune
import warnings
import torch

warnings.filterwarnings('ignore')

from common_config.common_config import parents_config

parents_config['test'] = False

from dual_attn_block_with_weather_v1.model import MainModel as model_1
from dual_attn_block_with_weather_v2.model import MainModel as model_2
from dual_attn_block_with_weather_v3.model import MainModel as model_3
from dual_attn_block_with_weather_v4.model import MainModel as model_4

from dual_attn_block_with_weather_v5.model import MainModel as v5
from dual_attn_block_with_weather_v5.config import parameter as v5_config

from util.running_util import easy_run, signal_config_run
from util.data_util import prepare_daloader

if __name__ == '__main__':
    path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
    #
    # best_config = {'common.block_num': 3,
    #                'common.block_len': 24,
    #                'common.prediction_horizon': 1,
    #                'input_encoder.hidden_size': 241,
    #                'input_encoder.directions': 1,
    #                'input_encoder.output_size': 87,
    #                'input_encoder.attn_hidden': 152,
    #                'input_encoder.num_layers': 1,
    #                'temporal_decoder.hidden_size': 82,
    #                'temporal_decoder.num_layers': 1,
    #                'weather.column_name': ['apparentTemperature', 'windSpeed', 'dewPoint'],
    #                'weather.out_size': 64,
    #                'weather.hidden_size': 256,
    #                'weather.input_size': 3,
    #                'weather.num_layers': 3,
    #                'running.lr': 0.0001,
    #                'running.batch_size': 4,
    #                'running.num_epoch': 600,
    #                'running.lrs_step_size': 2000,
    #                'running.max_grad_norm': 0.1,
    #                'running.gradient_accumulation_steps': 1,
    #                'running.reg1': False,
    #                'running.reg2': True,
    #                'running.reg_factor1': 0.0001,
    #                'running.reg_factor2': 0.0001,
    #                'running.data_succession': True}

    # config = {
    #     'gpu': 1 if torch.cuda.is_available() else 0,
    #     'test': False,
    #     'output_size': 1,
    #     # 编解码器中的公共参数
    #     "common.history_seq_len": 72,  # 可调整参数
    #     "common.block_num": 3,  # 长周期使用的天数 可调整参数
    #     "common.block_len": 24,  # 长时序使用的序列长度 可调整参数
    #     "common.prediction_horizon": 1,  # 预测的序列长度 可调整参数
    #
    #     # 运行时的一些参数
    #     "running.lr": 0.0001,
    #     "running.batch_size": 4,
    #     "running.lrs_step_size": 2000,  # 学习率下降步长
    #     "running.max_grad_norm": 0.1,  # 梯度剪切 最大梯度
    #     "running.gradient_accumulation_steps": 1,  # 梯度累计计算步长
    #     "running.reg1": False,  # L1 正则化
    #     "running.reg2": True,  # L2 正则化
    #     "running.reg_factor1": 1e-4,
    #     "running.reg_factor2": 1e-4,
    #     "running.data_succession": True,  # 样本数据是否连续
    #     'running.max_epoch': 500,
    #     'running.min_epoch': 100,
    #
    #     # 输入attention 编码器网络参数
    #     "input_encoder.hidden_size": 128,  # 可训练参数 跟decoder保持一致 lstm 的input_size 跟输入的long_days 保持一致
    #     "input_encoder.directions": 1,  # rnn方向
    #     "input_encoder.output_size": 216,  # 可训练参数
    #     "input_encoder.attn_hidden": 128,  # 可训练参数
    #     "input_encoder.num_layers": 1,
    #
    #     # 时序attention 和预测网络解码器参数
    #     "temporal_decoder.hidden_size": 128,
    #     "temporal_decoder.num_layers": 1,
    #
    #     # 天气因素编码参数
    #     "weather.column_name": ['apparentTemperature', 'windSpeed', 'dewPoint'],
    #     "weather.out_size": 160,  # 根据 编码融合的位置不同进行调整
    #     "weather.hidden_size": 128,
    #     "weather.input_size": 3,
    #     "weather.num_layers": 4,
    #
    # }
    dataloader = prepare_daloader(path,
                                  batch_size=v5_config['running.batch_size'],
                                  history_seq_len=v5_config['common.history_seq_len'])
    # signal_config_run(config=config,
    #                   run_model=model_4,
    #                   dataloader=dataloader,
    #                   ckp_path="./ray_results/model_4.ckpt")


    trainer, model, dataloader = signal_config_run(config=v5_config,
                                                   run_model=v5,
                                                   dataloader=dataloader,
                                                   ckp_path="./ray_results/v5.pt")
    trainer.predict(model, dataloader.train_dataloader())
'''

{'r2': 0.37111128437251284, 'evs': 0.3981284926523857, 'mae': 0.045785905000125394, 'mse': 0.003993691609489335, 'rmse': 0.06319566131855363}

'''
