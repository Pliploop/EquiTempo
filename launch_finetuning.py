from src.training.train import *
from src.data_loading.datasets import *
from torch.utils.tensorboard import SummaryWriter
from config.full import GlobalConfig

from src.finetuning.finetuning import finetune

import argparse





if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Process YAML config file")

    # Add an argument to accept the YAML config path
    parser.add_argument("--config_path", required=False, type=str, help="Path to the YAML config file", default=None)
    parser.add_argument("--dataset_name_list", nargs="+", default=None, required=False)
    parser.add_argument("--model_name", required=True)

    args = parser.parse_args()

    ## load config here or instanciate:
    globalconfig = GlobalConfig() ## or from yaml if load path exists
    if args.config_path is not None:
        print(f'loading config from {args.config_path}')
        globalconfig.from_yaml(args.config_path)

    finetune(model_name=args.model_name, dataset_name_list=args.dataset_name_list,global_config=globalconfig)