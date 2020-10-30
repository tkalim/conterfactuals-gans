import torch
import torchvision
from pathlib import Path
from .data.datasets import CelebADataset

CELEBA_DIR = "./data/celeba"
BATCH_SIZE = 20
ACTIATIONS_DIR = "./data/activations"
use_gpu = True if torch.cuda.is_available() else False


def predict(model, dataloader, model_weights=None):
    """
    predicts latent features of an celeba-like image
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

    return None


if __name__ == "__main__":
    dataset = torchvision.datasets.CelebA(
        root=CELEBA_DIR, split="all", target_type="attr", transform=None, download=True
    )
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
