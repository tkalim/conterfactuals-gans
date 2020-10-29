import torch
from pathlib import Path

use_gpu = True if torch.cuda.is_available() else False


def generate_actvations(model, concept, dataloader, dir):
    return None