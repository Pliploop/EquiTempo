import os
import time

import numpy as np
import torch
from config.dataset import MTATConfig
from config.full import GlobalConfig
from config.train import TrainConfig
from src.model.model import Siamese
from tqdm import tqdm

import wandb


class Trainer:

    def __init__(self, global_config = GlobalConfig()) -> None:
        self.global_config = global_config
        self.config = TrainConfig(dict = global_config.train_config)
        self.dataset_config = MTATConfig(dict = global_config.MTAT_config)
        if self.config.log_wandb:
            self.wandb_run = wandb.init(project="EquiTempo", config=global_config.to_dict())
            self.wandb_run_name = self.wandb_run.name
        else:
            self.wandb_run = None
            self.wandb_run_name = ""

    def init_model(self,path=None, test=False):
        device = self.config.device
        model = Siamese(filters=self.config.filters, dilations=self.config.dilations, dropout_rate=self.config.dropout_rate, output_dim=self.config.output_dim).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        it = 0
        if path is not None:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            it = checkpoint['it']
        if test:
            model.eval()
        return model,optimizer,scaler,it


    def save_model(self,loss, it, model, optimizer, scaler):
        os.makedirs(self.config.save_path, exist_ok=True)
        torch.save({
                'gen_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': loss,
                'it': it,
                }, self.config.save_path+f'/model_{self.wandb_run_name}_loss_{str(loss)[:6]}_it_{it}.pt')

    def save_config(self):
        self.global_config.save(self.config.save_path+f'/model_{self.wandb_run_name}.yml')


    def loss_function(self, c1, c2, alpha1, alpha2, eps=1e-7):
        c_ratio = c1/(c2+eps)
        alpha_ratio = alpha1/(alpha2+eps)
        return torch.abs(c_ratio-alpha_ratio).mean()


    def train_iteration(self, x1, x2, alpha1, alpha2, model, optimizer, scaler):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.mixed_precision):
            _,c1 = model(x1)
            _,c2 = model(x2)
            loss = self.loss_function(c1,c2,alpha1,alpha2)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()


    def update_lr(self, new_lr, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


    def train_loop(self, dataloader, model, optimizer, scaler, it=0, writer=None):
        self.save_config()
        dataloader_length = len(dataloader)
        if self.config.warmup:
            self.update_lr(1e-7, optimizer)
            target_lr = self.config.lr
            lr = 1e-7
        else:
            self.update_lr(self.config.lr, optimizer)
        model = model.to(self.config.device)
        model.train()
        try:
            counter = 0
            loss = 0.
            for epoch in range(self.config.epochs):
                bef = time.time()
                bef_loop = time.time()
                loss_list = []
                pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{self.config.epochs}', leave=True, total=dataloader_length)
                for batch_i, data in enumerate(pbar):
                    loss = self.train_iteration(data['audio_1'].to(self.config.device),data['audio_2'].to(self.config.device),data['rp_1'].to(self.config.device),data['rp_2'].to(self.config.device), model,optimizer,scaler)
                    if writer is not None:
                        writer.add_scalar('loss', loss, it)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)
                    if self.wandb_run is not None:
                        self.wandb_run.log({
                            "loss" : loss,
                            "learning_rate": optimizer.param_groups[0]['lr']
                        })
                    loss_list.append(loss)
                    counter += 1
                    it += 1
                    if self.config.warmup:
                        if epoch==0:
                            lr = lr + (1/dataloader_length)*target_lr
                            self.update_lr(lr, optimizer)

                    if batch_i%self.config.display_progress_every==0:
                        pbar.set_postfix({'Loss_sc': np.mean(loss_list[-counter:], axis=0),
                                            'Iter': it,
                                            'LR': optimizer.param_groups[0]['lr'],
                                            'Time/Iter': (time.time()-bef_loop)/self.config.display_progress_every})
                        bef_loop = time.time()
                self.save_model(np.mean(loss_list[-counter:], axis=0), it, model,optimizer,scaler)
                counter = 0
        except Exception as e:
            print(e)
        finally:
            self.save_model(np.mean(loss_list[-counter:], axis=0), it, model,optimizer,scaler)
            return it


