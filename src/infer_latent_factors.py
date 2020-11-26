from pathlib import Path

import numpy as np
import torch
from torch.utils import data

from .data.datasets import CelebADataset
from .models.vanilla_vae import VanillaVAE

CELEBA_DIR = "./data/celeba"
LATENT_FACTORS_DIR = "./data/generated/celeba-latent-factors"
BATCH_SIZE = 20
MODEL_STATE_DICT = "./models/AE.pt"
use_gpu = True if torch.cuda.is_available() else False


def predict(model, dataloader, model_weights=None):
    """
    predicts latent features of a celeba-like image
    """
    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights))

    model.eval()

    if use_gpu:
        model = model.to(device="cuda")
        print("CUDA available \U0001F600")
    else:
        print("CUDA not available \U0001F625")

    for ind_batch, sample_batched in enumerate(dataloader):
        # TODO:
        batch_images = sample_batched["image"]
        batch_filenames = sample_batched["filename"]
        if use_gpu:
            batch_images = batch_images.to(device="cuda")
        with torch.no_grad():
            output = model(batch_images)
            for idx, latent_vector in enumerate(output):
                filename = Path(batch_filenames[idx]).with_suffix(".npy")
                filepath = Path(LATENT_FACTORS_DIR) / filename
                np.save(filepath, latent_vector)

    return None


if __name__ == "__main__":
    dataset = CelebADataset(
        root=CELEBA_DIR,
        split="all",
        target_type="attr",
        transform=None,
        download=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    model = VanillaVAE()
    predict(model=model, dataloader=dataloader, model_weights=MODEL_STATE_DICT)
