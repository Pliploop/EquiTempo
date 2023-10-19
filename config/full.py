from config.dataset import MTATConfig, FinetuneDatasetConfig
from config.preprocessing import PreprocessingConfig
from config.train import TrainConfig
from config.finetune import FinetuneConfig
import yaml
from config.template import Config


class GlobalConfig(Config):
    def __init__(self, dict=None) -> None:
        super().__init__(dict=None)
        self.MTAT_config = MTATConfig().to_dict()
        self.finetune_dataset_config = FinetuneDatasetConfig().to_dict()
        self.preprocessing_config = PreprocessingConfig().to_dict()
        self.train_config = TrainConfig().to_dict()
        self.finetune_config = FinetuneConfig().to_dict()

    def from_yaml(self, path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        for k, v in data.items():
            setattr(self, k, v)
