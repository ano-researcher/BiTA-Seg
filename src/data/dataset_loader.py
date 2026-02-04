import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ImageMaskDataset(Dataset):
    def __init__(self, root, transforms=None, img_size=256):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")

        self.files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])

        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 128).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
