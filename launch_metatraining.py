import os

from config.full import GlobalConfig
from src.training.metatrain import meta_train

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    meta_train(GlobalConfig())
