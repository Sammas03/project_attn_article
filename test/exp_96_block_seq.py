'''
子序列长度实验  这里使用了 6 12 24
'''




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
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_block/Apt14_2015/{}_result.csv'

"""16"""
# path = r'../data/Apt16_2015_resample_hour_with_weather.xlsx'
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_block/Apt16_2015/{}_result.csv'

"""101"""
path = r'../data/Apt101_2015_resample_hour_with_weather.xlsx'
result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_block/Apt101_2015/{}_result.csv'

dataloader = prepare_daloader(path,
                              batch_size=parents_config['running.batch_size'],
                              history_seq_len=96)

def run_main_6():
    model = v5
    config = v5_config
    config['common.block_num'] = 12
    config['common.block_len'] = 6
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/block/MAIN6.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('MAIN6'))

def run_main_9():
    model = v5
    config = v5_config
    config['common.block_num'] = 8
    config['common.block_len'] = 9
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/block/MAIN9.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('MAIN9'))




def run_main_12():
    model = v5
    config = v5_config
    config['common.block_num'] = 8
    config['common.block_len'] = 12
    config['common.history_seq_len'] = 96
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/block_96/MAIN18.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('MAIN12'))



def run_main_24():
    model = v5
    config = v5_config
    config['common.block_num'] = 4
    config['common.block_len'] = 24
    config['common.history_seq_len'] = 96
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/block/MAIN24.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('MAIN24'))



def run_main_48():
    model = v5
    config = v5_config
    config['common.block_num'] = 2
    config['common.block_len'] = 48
    config['common.history_seq_len'] = 96
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/block_96/MAIN36.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('block96_MAIN48'))



def run_main_72():
    model = v5
    config = v5_config
    config['common.block_num'] = 4
    config['common.block_len'] = 24
    config['common.history_seq_len'] = 96
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/block/MAIN36.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('MAIN72'))




if __name__ == '__main__':
    # run_main_6()
    # run_main_9()
    # run_main_12()

    run_main_24()
    # run_main_48()
    # run_main_72()
# 完成：14    all
#      16    all  微调了 初始化 和dropout （去掉dropout gru效果最好）
#      101   all  proposal model 加上初始化效果比较好
#114 待调试
#
# 废弃：47，88