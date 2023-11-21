"""Evaluation."""

import argparse
import json
import os

import numpy as np
import torch

from config.evaluate import EvaluationConfig
from config.full import GlobalConfig
from src.data_loading.datasets import EvaluationDataset
from src.evaluation.metrics import compute_accuracy_1, compute_accuracy_2
from src.training.train import Trainer


def evaluate(model_name, dataset_name=None):
    config = EvaluationConfig(dict=GlobalConfig().evaluation_config)

    # override config if dataset list command line args provided
    if dataset_name:
        config.dataset_name = dataset_name

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
    )

    if model_name[-3:] != ".pt":
        model_name += ".pt"

    model_path = config.checkpoint_path + "/" + model_name
    model_path = config.checkpoint_path + "/" + model_name
    model, optimizer, scaler, it, epoch = Trainer().init_model(model_path)

    model.eval()

    dataset = EvaluationDataset(dataset_name=config.dataset_name, stretch=False)
    dataloader = dataset.create_dataloader()

    preds = []
    truths = []
    with torch.no_grad():
        for item_dict in dataloader:
            audio = item_dict["audio"].to(device)
            tempo = item_dict["tempo"]
            rf = item_dict["rf"]
            classification_pred, _ = model(audio)
            preds.append(np.argmax(classification_pred.cpu().numpy(), axis=1))
            truths.append((tempo * rf).cpu().numpy())

    # flatten arrays
    preds = [item for sublist in preds for item in sublist]
    truths = [item for sublist in truths for item in sublist]

    # create experiment dir from model name under results and save them
    experiment_dir = f"results/{model_name}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # save predictions, truths, results, and accuracies as .json files
    accuracy_1, results_1 = compute_accuracy_1(truths, preds)
    accuracy_2, results_2 = compute_accuracy_2(truths, preds)

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


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=None, required=False)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    dataset_name = args.dataset_name

    evaluate(args.model_name, dataset_name)
