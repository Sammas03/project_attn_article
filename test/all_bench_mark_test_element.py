
'''

对模型中的组件进行消融实验
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
from element_ablation.v5.fusion_v1.model import MainModel as v5_fusion_1
from element_ablation.v5.fusion_v2.model import MainModel as v5_fusion_2
from element_ablation.v5.AR.model import MainModel as delete_ar
from element_ablation.v5.without_weather.model import MainModel as delete_weather
# util
from util.running_util import easy_run, signal_config_run
from util.data_util import prepare_daloader
from util.predict_util import result_to_file

# supress warning
import warnings
warnings.filterwarnings('ignore')

"""apt 14"""
path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
result_path = r'/home/lhd/git_wsp/project_attn/ray_results/element_result/fusion/Apt14_2015/{}_result.csv'

"""16"""
# path = r'../data/Apt16_2015_resample_hour_with_weather.xlsx'
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_result/Apt16_2015/{}_result.csv'

"""101"""
# path = r'../data/Apt101_2015_resample_hour_with_weather.xlsx'
# result_path = r'/home/lhd/git_wsp/project_attn/ray_results/exp_result/Apt101_2015/{}_result.csv'

dataloader = prepare_daloader(path,
                              batch_size=parents_config['running.batch_size'],
                              history_seq_len=parents_config['common.history_seq_len'])


def run_main():
    model = v5
    config = v5_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/element_ablation/MAIN.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('MAIN'))


def run_fusion_1():
    model = v5_fusion_1
    config = v5_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/element_ablation/v5_fusion_1.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('v5_fusion_1'))





def run_fusion_2():
    model = v5_fusion_2
    config = v5_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/element_ablation/v5_fusion_2.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('v5_fusion_2'))




def run_fusion_2():
    model = v5_fusion_2
    config = v5_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/element_ablation/v5_fusion_2.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('v5_fusion_2'))




def run_delete_ar():
    model = delete_ar
    config = v5_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/element_ablation/delete_ar.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('delete_ar'))


def run_delete_weather():
    model = delete_weather
    config = v5_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/element_ablation/delete_weather.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('delete_ar'))

if __name__ == '__main__':
    # run_main()
    # run_fusion_1()
    # run_delete_ar()
    run_delete_weather()

# 完成：14    all
#      16    all  微调了 初始化 和dropout （去掉dropout gru效果最好）
#      101   all  proposal model 加上初始化效果比较好
#114 待调试
#
# 废弃：47，88

"""
16 gamma 0.94 ,init,no l2

101 gamma 0.92 init ,no l2 
gamma 0.94 下 without 0.47
"""