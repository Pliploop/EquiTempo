import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from src.evaluation.evaluation import evaluate
from src.training.train import Trainer

evaluate("filters_16_do_0.1_od_300/model_loss_0.0661_it_227856.pt", "gtzan")
