import os
import numpy as np

import pytorch_lightning as pl
import albumentations as A
import cv2

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from albumentations.pytorch import ToTensorV2


# Albumentations wrapper for ImageFolder
class AlbumentationsTransform:
    def __init__(self, alb_transform):
        self.alb_transform = alb_transform

    def __call__(self, img):
        img = np.array(img)  # PIL -> numpy
        augmented = self.alb_transform(image=img)
        return augmented["image"]

class LCCFASDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_transform = AlbumentationsTransform(A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.ISONoise(color_shift=(0.15,0.35), intensity=(0.1,0.5), p=0.05),
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2,0.2), p=0.125),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.CoarseDropout(num_holes_range=(3,6), hole_height_range=(10,20), hole_width_range=(10,20), fill="random_uniform", p=1.0),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.5931,0.4690,0.4229], std=[0.2471,0.2214,0.2157]),
            ToTensorV2()
        ]))

        test_transform = AlbumentationsTransform(A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.5931,0.4690,0.4229], std=[0.2471,0.2214,0.2157]),
            ToTensorV2()
        ]))

        self.train_data = ImageFolder(os.path.join(self.data_dir, "LCC_FASD_training"), transform=train_transform)
        self.val_data = ImageFolder(os.path.join(self.data_dir, "LCC_FASD_evaluation"), transform=test_transform)
        self.test_data = ImageFolder(os.path.join(self.data_dir, "LCC_FASD_development"), transform=test_transform)
        print(self.val_data.class_to_idx)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
