from src.training.train import *
from src.data_loading.datasets import *
from torch.utils.tensorboard import SummaryWriter
from config.full import GlobalConfig

import argparse





if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Process YAML config file")

    # Add an argument to accept the YAML config path
    parser.add_argument("--config_path", required=False, type=str, help="Path to the YAML config file", default=None)
    parser.add_argument("--resume_id", required=False, help="wandb run id to resume", default=None)
    parser.add_argument("--resume_checkpoint", required=False, help="checkpoint path to resume training from", default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--no-log', dest='feature', action='store_false')
    parser.add_argument('--resume_latest', default=None, required=False)
    

    args = parser.parse_args()
    
    globalconfig = GlobalConfig() ## or from yaml if load path exists
    if args.config_path is not None:
        print(f'loading config from {args.config_path}')
        globalconfig.from_yaml(args.config_path)

    
    if not args.resume_latest:
        config_path = args.config_path
        resume_checkpoint = args.resume_checkpoint
    else:
        
        config_path = os.path.join(args.resume_latest,'config.yml')
        resume_checkpoint = os.path.join(args.resume_latest,'latest.pt')

    ## load config here or instanciate:

    log = args.log
    trainer = Trainer(global_config=globalconfig, override_wandb=log, resume_id=args.resume_id)
    model, optimizer, scaler, it, epoch = trainer.init_model(path=resume_checkpoint)
    dataset = MTATDataset(global_config=globalconfig)
    train_dataloader,val_dataloader = dataset.create_dataloader()
    writer = SummaryWriter()

    trainer.train_loop(train_dataloader,val_dataloader, model, optimizer, scaler, writer=writer, it=it, epoch = epoch)