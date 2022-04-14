from ray import tune
import warnings
from util.mail_util import finished_mail
warnings.filterwarnings('ignore')

from common_config.common_config import parents_config

parents_config['test'] = False
from mlp_with_weather_tune.model import MlpModel
from mlp_with_weather_tune.config import parameter as mlp_config

from lstm_with_weather_tune.model import LstmModel
from lstm_with_weather_tune.config import parameter as lstm_config

from gru_with_weather_tune.model import GruModel
from gru_with_weather_tune.config import parameter as gru_config

from AE_MLP_tune.model import AeMlpModel
from AE_MLP_tune.config import parameter as aemlp_config

from AE_LSTM_tune.model import AeLstmModel
from AE_LSTM_tune.config import parameter as aelstm_config

from AE_GRU_tune.model import AeGruModel
from AE_GRU_tune.config import parameter as aegru_config
from util.running_util import easy_run, signal_config_run
from util.data_util import prepare_daloader
path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
if __name__ == '__main__':
    # path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
    #
    # mlp_config['input_size'] = mlp_config['common.history_seq_len'] * 4
    # result, dataloader = easy_run(data_path=path,
    #                               run_model=MlpModel,
    #                               saving_name='MLP_TEST',
    #                               config=mlp_config,
    #                               local_dir='./ray_results/apt14',
    #                               num_samples=1)
    #
    # lstm_config['input_size'] = 4
    # result, dataloader = easy_run(data_path=path,
    #                               run_model=LstmModel,
    #                               saving_name='LSTM_TEST',
    #                               config=lstm_config,
    #                               local_dir='./ray_results/apt14',
    #                               num_samples=5)

    # gru_config['input_size'] = 4
    # result, dataloader = easy_run(data_path=path,
    #                               run_model=GruModel,
    #                               saving_name='GRU_TEST',
    #                               config=gru_config,
    #                               num_samples=1)

    # aemlp_config['input_size'] = 24*4 #不需要
    # result, dataloader = easy_run(data_path=path,
    #                               run_model=AeMlpModel,
    #                               saving_name='AEMLP_TEST',
    #                               config=aemlp_config,
    #                               num_samples=1)

    # aelstm_config['input_size'] = 4
    # result, dataloader = easy_run(data_path=path,
    #                               run_model=AeLstmModel,
    #                               saving_name='AELSTM_TEST',
    #                               config=aelstm_config,
    #                               num_samples=1)

    # aegru_config['input_size'] = 4
    # result, dataloader = easy_run(data_path=path,
    #                               run_model=AeGruModel,
    #                               saving_name='AEGRU_TEST',
    #                               config=aegru_config,
    #                               local_dir='./ray_results',
    #                               num_samples=1)

    path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
    # mlp_config['mlp.layer1.hidden_num'] = tune.choice([512, 1024, 1152, 1440, 2048])
    # mlp_config['mlp.layer2.hidden_num'] = tune.choice([288, 512, 1024])
    # mlp_config['mlp.layer3.hidden_num'] = tune.randint(64, 288)
    # mlp_config['mlp.layer4.hidden_num'] = tune.randint(64, 128)
    # mlp_config['input_size'] = mlp_config['common.history_seq_len'] * 4
    # result, dataloader = easy_run(data_path=path,
    #                               run_model=MlpModel,
    #                               saving_name='MLP',
    #                               config=mlp_config,
    #                               local_dir='./ray_results/Apt14_2015/step1',
    #                               num_samples=100)
    # config = {'gpu': 1,
    #           'test': False,
    #           'output_size': 1,
    #           'common.block_num': 6,
    #           'common.block_len': 4,
    #           'common.history_seq_len': 72,
    #           'common.prediction_horizon': 1,
    #           'running.lr': 0.01,
    #           'running.batch_size': 4,
    #           'running.num_epoch': 1,
    #           'running.lrs_step_size': 2000,
    #           'running.max_grad_norm': 0.1,
    #           'running.gradient_accumulation_steps': 1,
    #           'running.reg1': False,
    #           'running.reg2': True,
    #           'running.reg_factor1': 0.0001,
    #           'running.reg_factor2': 0.0001,
    #           'running.data_succession': True,
    #           'running.max_epoch': 500,
    #           'running.min_epoch': 100,
    #           'lstm.hidden_num': 256,
    #           'lstm.num_layers': 4,
    #           'input_size': 4}
    # dataloader = prepare_daloader(path,
    #                               batch_size=config['running.batch_size'],
    #                               history_seq_len=config['common.history_seq_len'])
    # signal_config_run(config=config,
    #                   run_model=LstmModel,
    #                   dataloader=dataloader,
    #                   ckp_path="./ray_results/lstm.ckpt")
# {'r2': 0.4710003630101709, 'evs': 0.4718347678477276, 'mae': 0.04247411757565001, 'mse': 0.0033593565271102556, 'rmse': 0.05795995623799466}
# {'r2': 0.4480938584424785, 'evs': 0.4584191662216539, 'mae': 0.042866726027151046,
#                        'mse': 0.0035048218738742607, 'rmse': 0.0592015360769825}
    '''
    {'r2': 0.4379927049480966, 'evs': 0.4546009961083557, 'mae': 0.04307213240262994, 'mse': 0.0035689681861775843, 'rmse': 0.05974084186030177}
    
    '''

    config = {'gpu': 1,
              'test': False,
              'output_size': 1,
              'common.block_num': 6,
              'common.block_len': 4,
              'common.history_seq_len': 72,
              'common.prediction_horizon': 1,
              'running.lr': 0.0001,
              'running.batch_size': 4,
              'running.num_epoch': 1,
              'running.lrs_step_size': 2000,
              'running.max_grad_norm': 0.1,
              'running.gradient_accumulation_steps': 1,
              'running.reg1': False,
              'running.reg2': True,
              'running.reg_factor1': 0.0001,
              'running.reg_factor2': 0.0001,
              'running.data_succession': True,
              'running.max_epoch': 500,
              'running.min_epoch': 100,
              'gru.hidden_num': 256,
              'gru.num_layers': 4,
              'input_size': 4}
    dataloader = prepare_daloader(path,
                                  batch_size=config['running.batch_size'],
                                  history_seq_len=config['common.history_seq_len'])
    signal_config_run(config=config,
                      run_model=GruModel,
                      dataloader=dataloader,
                      ckp_path="./ray_results/gru.ckpt")

