import torch
import torch.nn as nn
from config.dataset import FinetuneConfig
from config.full import GlobalConfig
from src.model.model import Siamese
from src.data_loading.datasets import FinetuneDataset

model_path = "model.pt"
model = Siamese()
model.load_state_dict(torch.load(model_path))

# freeze up to head, so that only hat is trained
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


loss_function = nn.CrossEntropyLoss()
torch.optim.Adam(model.head.parameters(), lr=0.0001)

dataset = FinetuneDataset()
dataloader = dataset.create_dataloader()

for epoch in range(100):
    for i, dict in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(dict["audio"])
        loss = loss_function(outputs, dict["tempo"] * dict["rf"])
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{100}, Loss: {loss.item():.4f}")

# save in the same dir, add initials of datasets used
datasets_initials = "".join([dataset_name[:1] for dataset_name in dataset_list])
ft_model_path = model_path.replace(".pt", "{dataset_initials}.pt")
