import os
import pandas as pd
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from typing import Any, Callable, List, Optional, Union, Tuple


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
    as well as being able to select a subset of one attribute
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


class SmilingNotSmilingCelebADataset(CelebA):
    """
    slightly modified CelebA dataset to return smiling and
    not smiling images
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(self.__class__, self).__init__(
            root,
            split,
            target_type,
            transform,
            target_transform,
            download,
        )
        self.positive_class = "Smiling"

        attributes_list = list(
            pd.read_csv(
                Path(self.root) / "celeba" / "list_attr_celeba.txt",
                delim_whitespace=True,
                skiprows=1,
            ).columns
        )
        self.positive_class_idx = attributes_list.index(self.positive_class)

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

        if self.attr[index, self.positive_class_idx] == 1:
            return {
                "image": X,
                "filename": self.filename[index],
                "label_name": "Smiling",
                "label": 1,
            }
        # in the pytorch dataset -1 is mapped to 0
        else:
            return {
                "image": X,
                "filename": self.filename[index],
                "label_name": "Not Smiling",
                "label": 0,
            }


class ConceptsDataset(Dataset):
    """Attributes Concepts Dataset"""

    def __init__(self, attributes_csv_file, root_dir, concept, n_samples):
        """[summary]

        Args:
            attributes_csv_file ([type]): Location of the list_attributes csv
            root_dir ([type]): Path to dir containing the latent factors
        """
        self.attribtes = pd.read_csv(
            attributes_csv_file, delim_whitespace=True, skiprows=1
        )
        self.root_dir = root_dir
        self.concept = concept
        if n_samples <= len(self.attribtes):
            self.n_samples = n_samples
        else:
            print(f"Can't sample more than {len(self.attributes)}")
            raise ValueError
        self.samples = self.attribtes.groupby(concept, as_index=False).apply(
            lambda x: x.sample(n=self.n_samples)
        )
        self.samples = self.samples[concept]
        self.samples["latent_factor"] = self.samples.index.str.replace(".jpg", ".npy")
        self.samples.reset_index(drop=True, inplace=True)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int):
        latent_factor_path = (
            Path(self.root_dir) / self.samples.iloc[index]["latent_factor"]
        )
        latent_factor = np.load(latent_factor_path)
        label = self.samples.iloc[index][self.concept]

        return {"latent_factor": latent_factor, "label": label}
