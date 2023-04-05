from typing import Optional

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SkySegmentationDataset(Dataset):
    """Dataset class for sky segmentation task."""
    def __init__(
            self,
            image_dir: str,
            mask_dir: str,
            transform: Optional[transforms.Compose] = None
        ) -> None:
        """
        Initializes the SkySegmentationDataset class.

        Args:
            image_dir (str): Directory containing the input images.
            mask_dir (str): Directory containing the corresponding masks.
            transform (Optional[transforms.Compose]): PyTorch transforms to be applied on the images and masks.
        """

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the image and mask corresponding to the given index.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of image and corresponding mask.
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_seg.png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
