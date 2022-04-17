from ray import tune
import warnings
import torch

warnings.filterwarnings('ignore')

from common_config.common_config import parents_config

parents_config['test'] = False

from AE_cnn_attn_fuse.model import CnnAttnFuse
from AE_cnn_attn_fuse.config import parameter as caf_config

from AE_cnn_attn_fuse_v2.model import CnnAttnFuse as v2
from AE_cnn_attn_fuse_v2.config import parameter as v2_config

from AE_cnn_multi_encoder.model import CnnMultiHeader
from AE_cnn_multi_encoder.config import parameter as cmh_config

from util.running_util import easy_run, signal_config_run
from util.data_util import prepare_daloader

if __name__ == '__main__':
    path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'

    dataloader = prepare_daloader(path,
                                  batch_size=caf_config['running.batch_size'],
                                  history_seq_len=caf_config['common.history_seq_len'])

    caf_config["running.min_epoch"] = 35
    caf_config['running.lr'] = 0.0001
    signal_config_run(config=caf_config,
                      run_model=CnnAttnFuse,
                      dataloader=dataloader,
                      ckp_path="./ray_results/caf.pt")

    #
    # signal_config_run(config=cmh_config,
    #                   run_model=CnnMultiHeader,
    #                   dataloader=dataloader,
    #                   ckp_path="./ray_results/cmh.pt")
    # #
    # signal_config_run(config=v2_config,
    #                   run_model=v2,
    #                   dataloader=dataloader,
    #                   ckp_path="./ray_results/caf_v2.pt")
'''
{'r2': 0.4195044477756331, 'evs': 0.4202514757417378, 'mae': 0.0439856604953702, 'mse': 0.0036863759177983954, 'rmse': 0.060715532755616954}
{'r2': 0.4258069578707184, 'evs': 0.4266697748250734, 'mae': 0.04347836408033325, 'mse': 0.0036463524906641524, 'rmse': 0.06038503532055067}
{'r2': 0.4268868005423432, 'evs': 0.4278182686279325, 'mae': 0.04342855716293508, 'mse': 0.003639495063411808, 'rmse': 0.06032822774963481}

'''
# signal_config_run(config=caf_config,
#                   run_model=CnnAttnFuse,
#                   dataloader=dataloader,
#                   ckp_path="./ray_results/caf.pt")

'''
{'r2': 0.13764019581266407, 'evs': 0.17741294387977435, 'mae': 0.054217892115195974, 'mse': 0.005476325188801497, 'rmse': 0.07400219718901255}
{'r2': 0.15336042732066335, 'evs': 0.22416536410705257, 'mae': 0.05270756681284836, 'mse': 0.005376495512878492, 'rmse': 0.0733245900969006}
{'r2': 0.14174968242562092, 'evs': 0.22184918267260056, 'mae': 0.05470190621449046, 'mse': 0.005450228326514662, 'rmse': 0.0738256617072591}
{'r2': 0.19895485791311462, 'evs': 0.26768518608690783, 'mae': 0.05220660966549193, 'mse': 0.005086952879385963, 'rmse': 0.0713228776717959}
{'r2': 0.3615948174749093, 'evs': 0.3683346919502333, 'mae': 0.04682542176908283, 'mse': 0.004054124931087451, 'rmse': 0.0636720105783338}
{'r2': 0.3810066683082689, 'evs': 0.394063572333597, 'mae': 0.04518838761525861, 'mse': 0.00393085201511456, 'rmse': 0.06269650720027839}

'''
