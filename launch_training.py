from src.training.train import *
from src.data_loading.datasets import *
from torch.utils.tensorboard import SummaryWriter
from config.full import GlobalConfig

import wandb


## global config path with args

if __name__=="__main__":

    ## generate name for experiment

    ## load config here or instanciate:
    globalconfig = GlobalConfig() ## or from yaml if load path exists

    trainer = Trainer(global_config=globalconfig)
    model, optimizer, scaler, it = trainer.init_model()
    dataset = MTATDataset(global_config=globalconfig)
    dataloader = dataset.create_dataloader()
    writer = SummaryWriter()

    trainer.train_loop(dataloader, model, optimizer, scaler, writer=writer)