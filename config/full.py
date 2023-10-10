from config.dataset import MTATConfig
from config.preprocessing import PreprocessingConfig
from config.train import TrainConfig
from config.template import Config

class GlobalConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.MTAT_config = MTATConfig().to_dict()
        self.preprocessing_config = PreprocessingConfig().to_dict()
        self.train_config = TrainConfig().to_dict()
