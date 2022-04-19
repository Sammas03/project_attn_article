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


# util
from util.running_util import easy_run, signal_config_run
from util.data_util import prepare_daloader
from util.predict_util import result_to_file

path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
result_path = r'./exp_result/Apt14_2015_resample_hour_with_weather/{}.xlsx'
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
    config = gru_config
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
    model = AeMlpModel
    config = gru_config
    trainer, model, _ = signal_config_run(config=config,
                                          run_model=model,
                                          dataloader=dataloader,
                                          ckp_path="./ray_results/GRU.pt")
    result = trainer.predict(model, dataloader)
    result_to_file(result, result_path.format('GRU'))


