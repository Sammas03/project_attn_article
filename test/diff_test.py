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
from util.data_util import prepare_daloader, MyDataloder

if __name__ == '__main__':
    path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
    mydl = MyDataloder(path,
                       batch_size=caf_config['running.batch_size'],
                       history_seq_len=caf_config['common.history_seq_len'])
    dataloader = mydl.prepare_daloader()
    predict_dl = mydl.prepare_predict_daloader()
    caf_config['test']=False
    caf_config["running.min_epoch"] = 50
    caf_config['running.lr'] = 0.0001
    trainer, model, dataloader = signal_config_run(config=caf_config,
                                                   run_model=CnnAttnFuse,
                                                   dataloader=dataloader,
                                                   ckp_path="./ray_results/caf.pt")
    diff_result = trainer.predict(model, predict_dl.train_dataloader())
    from util.predict_util import predict_result_summary

    print(diff_result)
    real, pred = predict_result_summary(diff_result)
