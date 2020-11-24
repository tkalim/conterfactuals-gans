import time
from pathlib import Path

import torch
import torch.optim as optim
import torch.utils.data as data
from torch import nn
from torch.utils.data import dataloader
from torchvision import models
from torchvision import transforms as T

from data.datasets import SmilingNotSmilingCelebADataset

CELEBA_DIR = "./data/"
NUM_CLASSES = 1
CELEBA_DIMS = (218, 178)
MODEL_ACCEPTED_SIZE = 224
BATCH_SIZE = 20
CHECKPOINTS_DIR = "../models"
EPOCHS = 100
LEARNING_RATE = 0.01
SAVE_MODEL_EVERY_X_EPOCH = 10


def save_model(model, epoch, loss, save_dir):
    model_name = model.model_name
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
    learning_rate,
    model_weights=None,
    checkpoints_dir=CHECKPOINTS_DIR,
):
    cuda = torch.cuda.is_available()
    if cuda:
        print("GPU available")
        model = model.to(device="cuda")
    else:
        print("NO GPU")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for ind_batch, sample_batched in enumerate(dataloader):
            images = sample_batched["image"]
            groundtruths = sample_batched["label"]
            if cuda:
                images = images.to(device="cuda")
                groundtruths = groundtruths.to(device="cuda")

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, groundtruths.float().unsqueeze(1))

            loss.require_grad = True
            loss.backward()

            optimizer.step()

            if ind_batch % 100 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss
                    )
                )
        if epoch % SAVE_MODEL_EVERY_X_EPOCH == 0:

            save_model(
                model=model, epoch=epoch, loss=loss.item(), save_dir=checkpoints_dir
            )
            print(f"model saved to {str(checkpoints_dir)}")

    return None


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.Pad(padding=(CELEBA_DIMS[0] - CELEBA_DIMS[1], 0)),
            T.Resize(size=MODEL_ACCEPTED_SIZE),
            T.ToTensor(),
        ]
    )
    dataset = SmilingNotSmilingCelebADataset(
        root=CELEBA_DIR,
        split="train",
        target_type="attr",
        transform=transform,
        download=False,
    )
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    model = models.resnet101(pretrained=True)
    model.fc = nn.Linear(2048, NUM_CLASSES)
    train(
        model=model,
        dataloader=dataloader,
        epochs=EPOCHS,
        criterion=criterion,
        learning_rate=LEARNING_RATE,
        model_weights=None,
        checkpoints_dir=CHECKPOINTS_DIR,
    )
