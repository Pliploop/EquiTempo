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
    parser.add_argument("--model_name", required=False, default=None)
    parser.add_argument("--override_rf", required=False, default=None)
    parser.add_argument("--resume_id", required=False, default=None)
    parser.add_argument("--resume_checkpoint", required=False, default=None)
    

    args = parser.parse_args()

    ## load config here or instanciate:
    globalconfig = GlobalConfig() ## or from yaml if load path exists
    if args.config_path is not None:
        print(f'loading config from {args.config_path}')
        globalconfig.from_yaml(args.config_path)
    globalconfig.finetuning_config['finetuned_from'] = args.model_name
    
    if args.override_rf:
        override_rf = float(args.override_rf)
        globalconfig.preprocessing_config['rf'] = override_rf
        globalconfig.preprocessing_config['rf_range'] = override_rf
        
    finetune(model_name=args.model_name, dataset_name_list=args.dataset_name_list,global_config=globalconfig, resume_id = args.resume_id, resume_checkpoint=args.resume_checkpoint)