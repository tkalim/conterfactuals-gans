import time
from pathlib import Path
import numpy as np

import torch
from PIL import Image

use_gpu = True if torch.cuda.is_available() else False

MODEL = torch.hub.load(
    "facebookresearch/pytorch_GAN_zoo:hub",
    "PGAN",
    model_name="celebAHQ-512",
    pretrained=True,
    useGPU=use_gpu,
)

TRAIN_FOLDER = Path("../data/generated/train/")
TEST_FOLDER = Path("../data/generated/test/")


def generate_dataset(gan=MODEL, num_images=10, folder=TRAIN_FOLDER, batch_size=10):
    for i in range(num_images // batch_size):
        inputs, _ = gan.buildNoiseData(batch_size)
        with torch.no_grad():
            images = gan.test(inputs)
        for input_latent, image in zip(inputs, images):
            name = time.strftime("%Y%m%d-%H%M%S")
            image_path = folder / (name + ".jpeg")
            latent_path = folder / (name + ".pt")
            image = Image.fromarray(image.numpy())
            image.save(str(image_path))
            np.save(input_latent.numpy(), str(latent_path))
    return None
