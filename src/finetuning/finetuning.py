import argparse
import torch
import torch.nn as nn
from config.full import GlobalConfig
from config.finetune import FinetuneConfig
from src.model.model import Siamese
from src.data_loading.datasets import FinetuneDataset

config = FinetuneConfig(dict=GlobalConfig().finetune_config)

# argument parser argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_list", nargs="+", default=None, required=False)
parser.add_argument("--model_name", required=True)
args = parser.parse_args()
dataset_list = args.dataset_list
# override config if dataset list command line args provided
if not dataset_list:
    dataset_list = config.dataset_list

device = torch.device(
    "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
)
model_name = args.model_name
if model_name[-3:] != ".pt":
    model_name += ".pt"

model_path = config.checkpoint_path + "/" + model_name
model = Siamese()
model.load_state_dict(torch.load(model_path))

# freeze up to head, so that only hat is trained
for param in model.parameters():
    param.requires_grad = False
for param in model.hat.parameters():
    param.requires_grad = True

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

loss_function = nn.CrossEntropyLoss()
torch.optim.Adam(model.head.parameters(), lr=config.lr)

dataset = FinetuneDataset()
dataloader = dataset.create_dataloader()

for epoch in range(config.epochs):
    for i, dict in enumerate(dataloader):
        optimizer.zero_grad()
        # this currently restricts inference up to and including
        # the head to 1 batch...
        audio = dict["audio"].to(device)
        tempo = dict["tempo"].to(device)
        rf = dict["rf"].to(device)
        outputs = model(audio)
        loss = loss_function(outputs, tempo * rf)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{100}, Loss: {loss.item():.4f}")

# save in the same dir, add initials of datasets used
datasets_initials = "".join([dataset_name[:1] for dataset_name in dataset_list])
ft_model_path = model_path.replace(".pt", "{dataset_initials}.pt")
