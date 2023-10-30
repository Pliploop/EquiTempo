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

    model.train()

    loss_function = nn.CrossEntropyLoss()

    dataset = FinetuningDataset(dataset_name_list=dataset_name_list, stretch=False)
    dataloader = dataset.create_dataloader()
    print(len(dataloader))

    for epoch in range(config.epochs):
        for item_dict in dataloader:
            optimizer.zero_grad()
            audio = item_dict["audio"].to(device)
            # round and convert tempos to int
            tempo = item_dict["tempo"].to(device)
            rf = item_dict["rf"].to(device)
            final_tempos = torch.round(tempo * rf).long()
            classification_out, _ = model(audio)
            loss = loss_function(classification_out, final_tempos)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{100}, Loss: {loss.item():.4f}")

    # save in the same dir, add initials of datasets used
    datasets_initials = "".join(
        [dataset_name[:1] for dataset_name in dataset_name_list]
    )
    # make uppercase
    datasets_initials = datasets_initials.upper()
    ft_model_path = model_path.replace(".pt", f"_{datasets_initials}.pt")

    torch.save(model.state_dict(), ft_model_path)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_list", nargs="+", default=None, required=False)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    finetune(args.model_name, args.dataset_name_list)
