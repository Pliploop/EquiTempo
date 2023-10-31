import yaml

from config.dataset import EvaluationDatasetConfig, FinetuningDatasetConfig, MTATConfig
from config.evaluate import EvaluationConfig
from config.finetune import FinetuningConfig
from config.preprocessing import PreprocessingConfig
from config.template import Config
from config.train import TrainConfig


class GlobalConfig(Config):
    def __init__(self, dict=None) -> None:
        super().__init__(dict=None)

        self.MTAT_config = MTATConfig().to_dict()
        self.finetuning_dataset_config = FinetuningDatasetConfig().to_dict()
        self.evaluation_dataset_config = EvaluationDatasetConfig().to_dict()

        self.preprocessing_config = PreprocessingConfig().to_dict()
        self.train_config = TrainConfig().to_dict()
        self.finetuning_config = FinetuningConfig().to_dict()
        self.evaluation_config = EvaluationConfig().to_dict()

    def from_yaml(self, path):
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        for k, v in data.items():
            setattr(self, k, v)
