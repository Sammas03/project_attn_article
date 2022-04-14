from ray import tune
import warnings
from util.mail_util import finished_mail

from common_config.common_config import parents_config


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
from util.file_name_util import easy_add_time_suffix

parents_config['test'] = False
warnings.filterwarnings('ignore')
path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'

if __name__ == '__main__':
    aegru_config['en.hidden_num'] = 64
    aegru_config['en.num_layers'] = 3
    aegru_config['de.hidden_num'] = 128
    dataloader = prepare_daloader(path,
                                  batch_size=aegru_config['running.batch_size'],
                                  history_seq_len=aegru_config['common.history_seq_len'])
    signal_config_run(config=aegru_config,
                      run_model=AeGruModel,
                      dataloader=dataloader,
                      ckp_path=easy_add_time_suffix("./ray_results/aegru.ckpt"))
