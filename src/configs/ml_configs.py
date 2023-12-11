from dslib.nn.baseconfig import BaseNNConfig
from easydict import EasyDict as edict


class AMLConfig(BaseNNConfig):
    def __init__(self, match_base_and_overridden: bool = True) -> None:
        super().__init__(match_base_and_overridden)

    def get_overridden_config(self) -> edict:
        config = edict()
        # paths
        config.root_path = '/mnt/sata1/workdirs/market-research/predictor'
        config.paths = self.get_paths(config.root_path)
        config.model_filename_suffix = 'predictor'
        # layout
        config.config_description = self.__class__.__name__
        # common
        config.batch_size = 16
        config.inference_batch_size = 256
        config.num_epochs = 500
        config.tr_va_te_fractions = (0.70, 0.10, 0.20)
        config.debug = False
        config.scale_features = False  # False for vectors!
        config.scale_target = False
        config.left_data_bound_to_use = 0.0
        config.save_onnx = True
        config.compare_torch_and_onnx_outputs = False
        # optimizer and scheduler
        config.learning_rate = 1e-4
        config.learning_rate_min = 1e-6
        config.cosw_period = 10000
        # checkpoints
        config.checkpoint_to_load = 0
        # early stopping
        config.es_increasing_serie_length = 5
        config.es_constant_serie_length = 10
        # return
        return config


def init_ml_config(letter: str) -> BaseNNConfig:
    """
    Returns an instance of the selected ML config (`ml_config_obj`).

    `letter`:
        {'a', 'b', 'c', ...} according to a pipeline section and related flag.
    """
    if letter == 'a':
        return AMLConfig()
    else:
        raise ValueError('Unknown letter')
