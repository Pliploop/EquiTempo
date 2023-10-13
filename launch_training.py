from src.training.train import *
from src.data_loading.datasets import *
from torch.utils.tensorboard import SummaryWriter

trainer = Trainer()
model, optimizer, scaler, it = trainer.init_model()
dataset = MTATDataset()
dataloader = dataset.create_dataloader()
writer = SummaryWriter()

trainer.train_loop(dataloader, model, optimizer, scaler, writer=writer)