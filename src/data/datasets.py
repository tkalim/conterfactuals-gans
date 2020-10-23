from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
from torchvision import transforms


class GeneratedImagesDatasetTrain(Dataset):
    """Train dataset for the encoder"""

    def __init__(self, rooth_dir):
        self.root_dir = Path(root_dir)
        self.img_names = [
            x.name for x in self.root_dir.glob("**/*.jpeg") if x.is_file()
        ]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = root_dir / img_names[idx]
        gt_path = root_dir / (img_names[idx].split(".")[0] + ".npy")
        image = Image.open(str(image_path))
        transformation = transforms.ToTensor()
        image = transformation(image)
        gt = np.load(gt_path)
        sample = {"image": image, "gt": gt}

        return sample