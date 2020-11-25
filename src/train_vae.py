import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from models.vanilla_vae import VanillaVAE
from data.datasets import GeneratedImagesDatasetTrain

CHECKPOINTS_DIR = "../models/"
MODEL_NAME = "vanilla_vae"
LEARNING_RATE = 0.01
SAVE_MODEL_EVERY_X_EPOCH = 10
MODEL = VanillaVAE(in_channels=3, latent_dim=512)
BATCH_SIZE = 20
CRITERION = MODEL.encode_loss_function
EPOCHS = 10
TRAIN_DIR = Path("./data/generated/train")


def save_model(model, epoch, loss, save_dir, model_name):
    model_name = model_name
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"{timestr}_{model_name}_epoch_{epoch}_loss_{loss:03.3f}.pt"
    Path(save_dir).mkdir(exist_ok=True)
    file_path = Path(save_dir) / file_name
    torch.save(model.state_dict(), str(file_path))


def train(
    model,
    dataloader,
    epochs,
    criterion,
    model_weights=None,
    checkpoints_dir=CHECKPOINTS_DIR,
):

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA available")
    else:
        print("NO CUDA")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        for ind_batch, sample_batched in enumerate(dataloader):
            images = sample_batched["image"]
            groundtruths = sample_batched["groundtruth"]
            if cuda:
                images = images.to(device="cuda")
                groundtruths = groundtruths.to(device="cuda")

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, groundtruths)

            loss.require_grad = True
            loss.backward()

            optimizer.step()

            if ind_batch % 10 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss
                    )
                )
        if epoch % SAVE_MODEL_EVERY_X_EPOCH == 0 and epoch != 0:
            save_model(
                model=model, epoch=epoch, loss=loss.item(), save_dir=checkpoints_dir
            )
            print(f"model saved to {str(checkpoints_dir)}")


if __name__ == "__main__":
    model = MODEL
    dataset = GeneratedImagesDatasetTrain(root_dir=TRAIN_DIR)
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=4)
    train(
        model=MODEL,
        dataloader=dataloader,
        epochs=EPOCHS,
        criterion=CRITERION,
        checkpoints_dir=CHECKPOINTS_DIR,
    )
    print("training")
