from ray import tune
import warnings
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
from util.running_util import easy_run

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
    mlp_config['mlp.layer1.hidden_num'] = tune.choice([512, 1024, 1152, 1440, 2048])
    mlp_config['mlp.layer2.hidden_num'] = tune.choice([288, 512, 1024])
    mlp_config['mlp.layer3.hidden_num'] = tune.randint(64, 288)
    mlp_config['mlp.layer4.hidden_num'] = tune.randint(64, 128)
    mlp_config['input_size'] = mlp_config['common.history_seq_len'] * 4
    result, dataloader = easy_run(data_path=path,
                                  run_model=MlpModel,
                                  saving_name='MLP',
                                  config=mlp_config,
                                  local_dir='./ray_results/Apt14_2015/step1',
                                  num_samples=100)
