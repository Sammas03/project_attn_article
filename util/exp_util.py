from ray.tune import ExperimentAnalysis


def easy_load_exp(path):
    exp = ExperimentAnalysis(
        default_metric='v_loss',
        default_mode='min',
        experiment_checkpoint_path=path)
    return exp
