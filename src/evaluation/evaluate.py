"""Evaluation."""

import argparse
import json
import os

import torch

from config.evaluate import EvaluationConfig
from config.full import GlobalConfig
from src.data_loading.datasets import EvaluationDataset
from src.evaluation.metrics import accuracy_1, accuracy_2
from src.training.train import Trainer

config = EvaluationConfig(dict=GlobalConfig().evaluation_config)

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default=None, required=False)
parser.add_argument("--model_name", required=True)
args = parser.parse_args()
dataset_name = args.dataset_name
# override config if dataset list command line args provided
if dataset_name:
    config.dataset_name = dataset_name

device = torch.device(
    "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
)

model_name = args.model_name
if model_name[-3:] != ".pt":
    model_name += ".pt"

model_path = config.checkpoint_path + "/" + model_name
model_path = config.checkpoint_path + "/" + model_name
model, optimizer, scaler, it = Trainer().init_model(model_path)

model.eval()

dataset = EvaluationDataset(dataset_name=config.dataset_name)
dataloader = dataset.create_dataloader()

preds = []
truths = []
with torch.no_grad():
    for i, item_dict in enumerate(dataloader):
        audio = item_dict["audio"].to(device)
        tempo = item_dict["tempo"]
        rf = item_dict["rf"]
        classification_pred, regression_pred = model(audio)
        preds.append(classification_pred)
        truths.append(tempo * rf)

# create experiment dir from model name under results and save them
experiment_dir = f"results/{model_name}"
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# save predictions, truths, results, and accuracies as .json files
preds = torch.cat(preds).cpu().numpy()
truths = torch.cat(truths).cpu().numpy()

results_1, accuracy_1 = accuracy_1(truths, preds)
results_2, accuracy_2 = accuracy_2(truths, preds)

results = {
    "preds": preds,
    "truths": truths,
    "accuracy_1": accuracy_1,
    "accuracy_2": accuracy_2,
}

print(f"Accuracy 1: {accuracy_1}")
print(f"Accuracy 2: {accuracy_2}")

with open("{experiment_dir}/results.json", "w") as f:
    json.dump(results, f)
