import argparse
import sys

import torch
import torch.nn as nn

from config.finetune import FinetuningConfig
from config.full import GlobalConfig
from src.data_loading.datasets import FinetuningDataset
from src.training.train import Trainer


def finetune(model_name, dataset_name_list=None):
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
    model, optimizer, scaler, it = Trainer().init_model(model_path)

    # freeze up to head, so that only hat is trained
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hat.parameters():
        param.requires_grad = True

    model.float()
    model.train()

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_function = nn.CrossEntropyLoss()
    torch.optim.Adam(model.head.parameters(), lr=config.lr)

    dataset = FinetuningDataset(dataset_name_list=dataset_name_list, stretch=True)
    dataloader = dataset.create_dataloader()

    for epoch in range(config.epochs):
        for item_dict in dataloader:
            optimizer.zero_grad()
            # this currently restricts inference up to and including
            # the head to 1 batch...
            audio = item_dict["audio"].to(device)
            tempo = item_dict["tempo"].to(device)
            rf = item_dict["rf"].to(device)
            classification_out, _ = model(audio)
            loss = loss_function(classification_out, tempo * rf)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{100}, Loss: {loss.item():.4f}")

    # save in the same dir, add initials of datasets used
    datasets_initials = "".join(
        [dataset_name[:1] for dataset_name in dataset_name_list]
    )
    ft_model_path = model_path.replace(".pt", f"{datasets_initials}.pt")

    torch.save(model.state_dict(), ft_model_path)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_list", nargs="+", default=None, required=False)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    finetune(args.model_name, args.dataset_name_list)
