from config.dataset import MTATConfig
from config.preprocessing import PreprocessingConfig
from config.train import TrainConfig
from config.template import Config

class GlobalConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.MTAT_config = MTATConfig()
        self.preprocessing_config = PreprocessingConfig()
        self.train_config = TrainConfig()

    def make_global_config(self):
        return {
            "MTAT_config" : self.MTAT_config.to_dict(),
            "preprocessing_config" : self.preprocessing_config.to_dict(),
            "train_config" : self.train_config.to_dict()
        }