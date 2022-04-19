

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

from CNN_LSTM.model import CnnLstmModel
from CNN_LSTM.config import parameter as cnnlstm_config

from dual_attn_block_with_weather_v5.model import MainModel as v5
from dual_attn_block_with_weather_v5.config import parameter as v5_config

# util
from util.running_util import easy_run, signal_config_run
from util.data_util import prepare_daloader
from util.predict_util import result_to_file

# supress warning
import warnings
warnings.filterwarnings('ignore')

"""apt 14"""
# path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_result/Apt14_2015/{}_result.csv'

"""apt 47"""
# path = r'../data/Apt47_2015_resample_hour_with_weather.xlsx'
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_result/Apt47_2015/{}_result.csv'


"""88"""
# path = r'../data/Apt88_2015_resample_hour_with_weather.xlsx'
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_result/Apt88_2015/{}_result.csv'


"""114"""
# path = r'../data/Apt114_2015_resample_hour_with_weather.xlsx'
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_result/Apt114_2015/{}_result.csv'


"""16"""
# path = r'../data/Apt16_2015_resample_hour_with_weather.xlsx'
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_result/Apt16_2015/{}_result.csv'

"""101"""
path = r'../data/Apt101_2015_resample_hour_with_weather.xlsx'
result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_result/Apt101_2015/{}_result.csv'

dataloader = prepare_daloader(path,
                              batch_size=parents_config['running.batch_size'],
                              history_seq_len=parents_config['common.history_seq_len'])


def run_mlp():
    model = MlpModel
    config = mlp_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/mlp.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('MLP'))


def run_lstm():
    model = LstmModel
    config = lstm_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/LSTM.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('LSTM'))



def run_gru():
    model = GruModel
    config = gru_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/GRU.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('GRU'))


def run_aemlp():
    model = AeMlpModel
    config = aemlp_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/AEMLP.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('AEMLP'))


def run_cnn_lstm():
    model = CnnLstmModel
    config = cnnlstm_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/CNNLSTM.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('CNNLSTM'))


# def run_cnn_gru():
#     pass


def run_main():
    model = v5
    config = v5_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/MAIN.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('MAIN'))



if __name__ == '__main__':
    # run_mlp()
    run_aemlp()
    # run_lstm()
    # run_gru()
    # run_cnn_lstm()
    # run_main()

# 完成：14    all
#      16    all  微调了 初始化 和dropout （去掉dropout gru效果最好）
#      101   all  proposal model 加上初始化效果比较好
#114 待调试
#
# 废弃：47，88