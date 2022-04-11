

from mlp_with_weather_tune.model import MlpModel as BasicModel
from mlp_with_weather_tune.config import parameter
from util.running_util import easy_run

if __name__ == '__main__':
    parameter['mlp.input_size'] = 24*4
    path = r'../data/Apt14_2015_resample_hour_with_weather.xlsx'
    result, dataloader = easy_run(data_path=path,
                                  run_model=BasicModel,
                                  saving_name='MLP_TEST',
                                  config=parameter,
                                  num_samples=1)
