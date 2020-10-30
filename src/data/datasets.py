import os
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA


class GeneratedImagesDatasetTrain(Dataset):
    """Train dataset for the encoder"""

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.img_names = [
            x.name for x in self.root_dir.glob("**/*.jpeg") if x.is_file()
        ]
        self.transformation = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.root_dir / self.img_names[idx]
        gt_path = self.root_dir / (self.img_names[idx].split(".")[0] + ".npy")
        image = Image.open(str(image_path))
        image = self.transformation(image)
        gt = np.load(gt_path)
        sample = {"image": image, "groundtruth": gt}

        return sample


class CelebADataset(CelebA):
    """
    slightly modified CelebA dataset to return the filename with the image
    """

    def __getitem__(self, index):
        X = PIL.Image.open(
            os.path.join(
                self.root, self.base_folder, "img_align_celeba", self.filename[index]
            )
        )

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError('Target type "{}" is not recognized.'.format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return {"image": X, "filename": self.filename[index], "target": target}
