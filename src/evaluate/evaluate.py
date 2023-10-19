import argparse
import torch
import torch.nn as nn
from config.full import GlobalConfig
from config.evaluate import EvaluateConfig
from src.model.model import Siamese
from src.data_loading.datasets import EvalDataset
