from src.training.train import *
from src.data_loading.datasets import *
from torch.utils.tensorboard import SummaryWriter
from config.full import GlobalConfig

import argparse





if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Process YAML config file")

    # Add an argument to accept the YAML config path
    parser.add_argument("--config_path", required=False, type=str, help="Path to the YAML config file", default=None)

    args = parser.parse_args()

    ## load config here or instanciate:
    globalconfig = GlobalConfig() ## or from yaml if load path exists
    if args.config_path is not None:
        print(f'loading config from {args.config_path}')
        globalconfig.from_yaml(args.config_path)

    trainer = Trainer(global_config=globalconfig, override_wandb=False)
    model, optimizer, scaler, it = trainer.init_model()
    dataset = MTATDataset(global_config=globalconfig)
    dataloader = dataset.create_dataloader()
    writer = SummaryWriter()

    trainer.train_loop(dataloader, model, optimizer, scaler, writer=writer)