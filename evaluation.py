"""Evaluation."""

import argparse
import json
import os
import pandas as pd

import numpy as np
import torch

from config.evaluate import EvaluationConfig
from config.full import GlobalConfig
from src.data_loading.datasets import EvaluationDataset
from src.evaluation.metrics import compute_accuracy_1, compute_accuracy_2
from src.training.train import Trainer
from src.model.model import Siamese
from tqdm import tqdm
import torch.nn.functional as F


def evaluate(model_name, dataset_name=None):
    config = EvaluationConfig(dict=GlobalConfig().evaluation_config)

    # override config if dataset list command line args provided
    if dataset_name:
        config.dataset_name = dataset_name

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )
    print("device", device)

    if model_name[-3:] != ".pt":
        model_name += ".pt"

    model_path = model_name
    model_path = model_name
    model, optimizer, scaler, it, epoch = Trainer(override_wandb=False).init_model(
        model_path
    )
    # model = Siamese().to(device)
    # model.load_state_dict(torch.load(model_path,map_location=device)['gen_state_dict'])

    model.eval()

    dataset = EvaluationDataset(dataset_name=config.dataset_name, stretch=False)
    dataloader = dataset.create_dataloader()

    preds = []
    truths = []
    with torch.no_grad():
        for item_dict in tqdm(dataloader):
            audio = item_dict["audio"].squeeze(0).to(device)
            tempo = item_dict["tempo"].squeeze().to(device)
            rf = item_dict["rf"].to(device)
            classification_pred, _ = model(audio)  ## logits of shape [audio_split, 300]
            classification_pred = F.softmax(
                classification_pred, dim=1
            )  ## softmax of shape [audio_split, 300
            classification_pred = torch.mean(
                classification_pred, dim=0
            )  ## mean of shape [300]

            preds.append(torch.argmax(classification_pred).item())
            truths.append((tempo * rf).cpu().numpy()[0][0])
            # print(f"pred: {torch.argmax(classification_pred).item()}, truth: {(tempo * rf).cpu().numpy()[0][0]}")

    # flatten arrays
    # preds = [item for sublist in preds for item in sublist]
    # truths = [item for sublist in truths for item in sublist]

    # create experiment dir from model name under results and save them
    experiment_dir = f"results/{model_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # save predictions, truths, results, and accuracies as .json files
    accuracy_1, results_1 = compute_accuracy_1(truths, preds, tol=0.04)
    accuracy_2, results_2 = compute_accuracy_2(truths, preds, tol=0.04)

    results = {
        # "preds": preds,
        # "truths": truths,
        "accuracy_1": f"{accuracy_1:.4f}",
        "accuracy_2": f"{accuracy_2:.4f}",
    }

    print(f"Accuracy 1: {accuracy_1}")
    print(f"Accuracy 2: {accuracy_2}")

    with open(f"{experiment_dir}/results.json", "w") as f:
        json.dump(results, f)

    return accuracy_1, accuracy_2


if __name__ == "__main__":
    # argument parser
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_name", default=None, required=False)
    # parser.add_argument("--model_name", required=True)
    # args = parser.parse_args()
    # dataset_name = args.dataset_name

    # evaluate(args.model_name, dataset_name)

    # dataset_names = ['giantsteps','gtzan','hainsworth']
    dataset_names = ["hainsworth"]
    augmentations = ["off", "on"]
    model_names = [
        "checkpoints/fine_tune/model__finetune_GTZ_GIA/it_9256_loss_2.6955_acc10.8_acc20.93.pt"
    ]

    # model_names = [
    #     [['checkpoints/fine_tune/giantsteps/augm_off/ratio_0.1/model_eternal-donkey-39_finetune_HAI_GTZ/latest.pt',
    #     'checkpoints/fine_tune/giantsteps/augm_off/ratio_0.2/model_eternal-donkey-39_finetune_finetune_HAI_GTZ/latest.pt',
    #     'checkpoints/fine_tune/giantsteps/augm_off/ratio_0.3/model_eternal-donkey-39_finetune_finetune_finetune_HAI_GTZ/latest.pt',
    #     'checkpoints/fine_tune/giantsteps/augm_off/ratio_0.4/model_eternal-donkey-39_finetune_finetune_finetune_finetune_HAI_GTZ/latest.pt'],
    #     ['checkpoints/fine_tune/giantsteps/augm_on/ratio_0.1/model_mild-wind-38_finetune_HAI_GTZ/latest.pt',
    #      'checkpoints/fine_tune/giantsteps/augm_on/ratio_0.2/model_mild-wind-38_finetune_finetune_HAI_GTZ/latest.pt',
    #      'checkpoints/fine_tune/giantsteps/augm_on/ratio_0.3/model_mild-wind-38_finetune_finetune_finetune_HAI_GTZ/latest.pt',
    #      'checkpoints/fine_tune/giantsteps/augm_on/ratio_0.4/model_mild-wind-38_finetune_finetune_finetune_finetune_HAI_GTZ/latest.pt']],
    #     [['checkpoints/fine_tune/gtzan/augm_off/ratio_0.1/model_glorious-disco-36_finetune_HAI_GIA/latest.pt',
    #     'checkpoints/fine_tune/gtzan/augm_off/ratio_0.2/model_glorious-disco-36_finetune_finetune_HAI_GIA/latest.pt',
    #     'checkpoints/fine_tune/gtzan/augm_off/ratio_0.3/model_glorious-disco-36_finetune_finetune_finetune_HAI_GIA/latest.pt',
    #     'checkpoints/fine_tune/gtzan/augm_off/ratio_0.4/model_glorious-disco-36_finetune_finetune_finetune_finetune_HAI_GIA/latest.pt'],
    #     ['checkpoints/fine_tune/gtzan/augm_on/ratio_0.1/model_quiet-resonance-35_finetune_HAI_GIA/latest.pt',
    #      'checkpoints/fine_tune/gtzan/augm_on/ratio_0.2/model_quiet-resonance-35_finetune_finetune_HAI_GIA/latest.pt',
    #      'checkpoints/fine_tune/gtzan/augm_on/ratio_0.3/model_quiet-resonance-35_finetune_finetune_finetune_HAI_GIA/latest.pt',
    #      'checkpoints/fine_tune/gtzan/augm_on/ratio_0.4/model_quiet-resonance-35_finetune_finetune_finetune_finetune_HAI_GIA/latest.pt']],
    #     [['checkpoints/fine_tune/hainsworth/augm_off/ratio_0.1/model_blooming-frost-28_finetune_GTZ_GIA/latest.pt',
    #     'checkpoints/fine_tune/hainsworth/augm_off/ratio_0.2/model_blooming-frost-28_finetune_finetune_GTZ_GIA/latest.pt',
    #     'checkpoints/fine_tune/hainsworth/augm_off/ratio_0.3/model_blooming-frost-28_finetune_finetune_finetune_GTZ_GIA/latest.pt',
    #     'checkpoints/fine_tune/hainsworth/augm_off/ratio_0.4/model_blooming-frost-28_finetune_finetune_finetune_finetune_GTZ_GIA/latest.pt'],
    #     ['checkpoints/fine_tune/hainsworth/augm_on/ratio_0.1/model_youthful-pond-27_finetune_GTZ_GIA/latest.pt',
    #      'checkpoints/fine_tune/hainsworth/augm_on/ratio_0.2/model_youthful-pond-27_finetune_finetune_GTZ_GIA/latest.pt',
    #      'checkpoints/fine_tune/hainsworth/augm_on/ratio_0.3/model_youthful-pond-27_finetune_finetune_finetune_GTZ_GIA/latest.pt',
    #      'checkpoints/fine_tune/hainsworth/augm_on/ratio_0.4/model_youthful-pond-27_finetune_finetune_finetune_finetune_GTZ_GIA/latest.pt']]
    # ]

    records = []

    for i, dataset in enumerate(dataset_names):
        dataset_name = dataset
        for j, augmentation in enumerate(augmentations):
            augmentation = augmentation
            for k, model in enumerate(model_names):
                model_name = model
                print(
                    f"Dataset: {dataset_name}, Augmentation: {augmentation}, Model: {model_name}"
                )
                acc1, acc2 = evaluate(model_name, dataset_name)
                records.append(
                    {
                        "dataset": dataset_name,
                        "augmentation": augmentation,
                        "model": model_name,
                        "acc1": acc1,
                        "acc2": acc2,
                    }
                )
                print(records)

    # save records as pandas dataframe to csv but also a a json file
    # df = pd.DataFrame.from_records(records)
    # df.to_csv("results.csv")
    # df.to_json("results.json")

    # print(df)