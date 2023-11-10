import argparse
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

from config.finetune import FinetuningConfig
from config.full import GlobalConfig
from src.data_loading.datasets import FinetuningDataset
from src.training.train import Trainer
import time
import numpy as np
from src.evaluation.metrics import compute_accuracy_1, compute_accuracy_2
import os
from src.finetuning.finetune_loss import XentBoeck

import wandb


def finetune(model_name, dataset_name_list=None, global_config =None):
    if global_config:
        config = FinetuningConfig(dict=global_config.finetuning_config) 
    else:
        config = FinetuningConfig(dict=GlobalConfig().finetuning_config)

    # use config arg if not provided in function call
    if not dataset_name_list:
        dataset_name_list = config.dataset_name_list

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    if model_name[-3:] != ".pt":
        model_name += ".pt"

    model_path = config.checkpoint_path + "/" + model_name
    trainer = Trainer()
    model, _, _, _ = Trainer().init_model(model_path)
    optimizer = torch.optim.Adam(model.parameters(),lr = config.lr)
    if config.wandb_log:
        
            wandb_run = wandb.init(project="EquiTempo", config=global_config.to_dict())
            wandb_run.name = wandb_run.name + '_finetune'
            wandb.watch(models=model,log='all', log_freq=10)
    else:    
            wandb_run = None
            wandb_run_name = ""

    model.train()

    # loss_function = nn.CrossEntropyLoss()
    loss_function = XentBoeck(device=device)
    dataset = FinetuningDataset(dataset_name_list=dataset_name_list, stretch=config.stretch)
    dataloader = dataset.create_dataloader()
    
    counter = 0
    loss = 0.
    it = 0
    
    first_run = True
            
    for epoch in range(config.epochs):
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False
        )
        for item_dict in progress_bar:
            optimizer.zero_grad()

            audio = item_dict["audio"].to(device)
            tempo = item_dict["tempo"].to(device)
            rf = item_dict["rf"].to(device)
            # round and convert tempos to int
            final_tempos = torch.round(tempo * rf).long()

            classification_out, _ = model(audio)
            preds = np.argmax(classification_out.detach().cpu().numpy(), axis=1)
            truth = (final_tempos).detach().cpu().numpy()
            
            
            
            if first_run:
                print(f" preds : {preds}")
                print(f"truth: {truth}")
                print(f"rf: {rf}")
                print()
                
            
            if it%20==0:
                print(classification_out.shape)
                print(final_tempos.shape)
                print(f" preds : {preds}")
                print(f"truth: {truth}")
            
            acc_1 = compute_accuracy_1(truth.tolist(), preds.tolist())
            acc_2 = compute_accuracy_2(truth.tolist(), preds.tolist())
            loss = loss_function(classification_out, final_tempos)
            
            if wandb_run is not None:
                        wandb_run.log({
                            "loss" : loss,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "acc_1" : acc_1[0],
                            "acc_2" : acc_2[0]
                        }, step = it)
            loss.backward()
            optimizer.step()
            counter += 1
            it += 1

            progress_bar.set_postfix(loss=loss.item())
            
            first_run = False

        print(f"Epoch {epoch+1}/{100}, Loss: {loss.item():.4f}")

    # save in the same dir, add initials of datasets used
    datasets_initials = "_".join(
        [dataset_name[:3] for dataset_name in dataset_name_list]
    )
    # make uppercase
    datasets_initials = datasets_initials.upper()
    ft_model_path = model_path.replace(".pt", f"_{datasets_initials}.pt")

    # torch.save(model.state_dict(), ft_model_path)
    save_model(config,loss,it,model,optimizer,wandb_run,acc_1, acc_2, datasets_initials)
    
def save_model(config, loss, it, model, optimizer, wandb_run, acc_1,acc_2, datasets_initials):
        os.makedirs(config.save_path, exist_ok=True)
        torch.save({
                'gen_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'it': it,
                }, config.save_path+f'/model_{wandb_run.name}_finetune_{datasets_initials}_loss_{str(loss)[:6]}_acc1{acc_1}_acc2{acc_2}_it_{it}.pt')

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_list", nargs="+", default=None, required=False)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    finetune(args.model_name, args.dataset_name_list)
