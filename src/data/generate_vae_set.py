from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as Transforms
import torchvision.utils as vutils
from PIL import Image

use_gpu = True if torch.cuda.is_available() else False
OUT_SIZE = 128

MODEL = torch.hub.load(
    "facebookresearch/pytorch_GAN_zoo:hub",
    "PGAN",
    model_name="celeba",
    pretrained=True,
    useGPU=use_gpu,
)

TRAIN_FOLDER = Path("../../data/generated/train")
TEST_FOLDER = Path("../../data/generated/test/")
NUM_IMAGES = 10000
BATCH_SIZE = 100


def resize_tensor(data, out_size_image):

    out_data_size = (
        data.size()[0],
        out_size_image[0],
        out_size_image[1],
    )

    outdata = torch.empty(out_data_size)
    data = torch.clamp(data, min=-1, max=1)

    interpolationMode = 0
    if out_size_image[0] < data.size()[0] and out_size_image[1] < data.size()[1]:
        interpolationMode = 2

    transform = Transforms.Compose(
        [
            Transforms.Normalize((-1.0, -1.0, -1.0), (2, 2, 2)),
            Transforms.ToPILImage(),
            Transforms.Resize(out_size_image, interpolation=interpolationMode),
            Transforms.ToTensor(),
        ]
    )

    # for img in range(out_data_size[0]):
    #    outdata[img] = transform(data[img])
    outdata = transform(data)

    return outdata


def save_tensor(data, out_size_image, path):
    outdata = resize_tensor(data, out_size_image)
    vutils.save_image(outdata, path)


def generate_dataset(
    gan=MODEL, num_images=10, folder=TRAIN_FOLDER, batch_size=BATCH_SIZE
):
    for i in range(num_images // batch_size):
        inputs, _ = gan.buildNoiseData(batch_size)
        with torch.no_grad():
            generated_images = gan.test(inputs)
        # TODO: check if inputs match images
        for input_latent, image in zip(inputs, generated_images):
            name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            image_path = folder / (name + ".jpeg")
            latent_path = folder / (name + ".npy")
            # image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
            # image.save(str(image_path))
            np.save(str(latent_path), input_latent.numpy())
            save_tensor(image, (OUT_SIZE, OUT_SIZE), image_path)
            print(f"{image_path.name} created")
        print(f"Batch {i+1}/{num_images//batch_size} done!")
    return None


if __name__ == "__main__":
    generate_dataset(
        gan=MODEL, num_images=NUM_IMAGES, folder=TRAIN_FOLDER, batch_size=10
    )
