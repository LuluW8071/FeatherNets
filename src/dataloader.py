import numpy as np
import pytorch_lightning as pl
import albumentations as A
import cv2

from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from dataset import LCCFASDDataset


# -------------------------------
# Albumentations wrapper
# -------------------------------
class AlbumentationsTransform:
    def __init__(self, alb_transform):
        self.alb_transform = alb_transform

    def __call__(self, image):
        """
        Always expects image in HWC numpy format.
        Returns tensor [C,H,W].
        """
        result = self.alb_transform(image=image)
        return result["image"]
    

# -------------------------------
# Depth Transform wrapper
# -------------------------------
class DepthTransform:
    def __init__(self, alb_transform):
        self.alb_transform = alb_transform

    def __call__(self, depth):
        """
        Ensures depth is single-channel HWC numpy before ToTensorV2.
        Returns tensor [1,H,W].
        """
        if depth.ndim == 2:                             # [H,W]
            depth = np.expand_dims(depth, -1)           # [H,W,1]
        elif depth.ndim == 3 and depth.shape[2] == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            depth = np.expand_dims(depth, -1)
        depth = depth.astype(np.float32) / 255.0        # Normalize to [0,1]

        result = self.alb_transform(image=depth)
        out = result["image"]  # [1,H,W]
        return out
    

# -------------------------------
# DataModule
# -------------------------------
class SpoofDataModule(pl.LightningDataModule):
    def __init__(self, lcc_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.base_dir = lcc_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # RGB transforms
        train_transform = AlbumentationsTransform(A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.Rotate(limit=15, p=0.2),
            A.ISONoise(color_shift=(0.15,0.35), intensity=(0.1,0.5), p=0.05),
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2,0.2), p=0.125),
            A.MotionBlur(blur_limit=3, p=0.2),
            # A.CoarseDropout(num_holes_range=(3,6),
            #                 hole_height_range=(10,20),
            #                 hole_width_range=(10,20),
            #                 fill_value=0, p=0.125),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.4623, 0.4126, 0.4097],
                        std=[0.3034, 0.2961, 0.2941]),
            ToTensorV2()
        ]))

        test_transform = AlbumentationsTransform(A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.4623, 0.4126, 0.4097],
                        std=[0.3034, 0.2961, 0.2941]),
            ToTensorV2()
        ]))

        # Depth transforms
        depth_transform = DepthTransform(A.Compose([
            A.Resize(14, 14, interpolation=cv2.INTER_CUBIC),
            ToTensorV2()
        ]))

        # Datasets
        self.train_data = LCCFASDDataset(self.base_dir, split="LCC_FASD_training",
                                         transform=train_transform, map_transform=depth_transform)
        self.val_data = LCCFASDDataset(self.base_dir, split="LCC_FASD_evaluation",
                                       transform=test_transform, map_transform=depth_transform)
        self.test_data = LCCFASDDataset(self.base_dir, split="LCC_FASD_development",
                                        transform=test_transform, map_transform=depth_transform)

        # Debug dataset sizes
        print("Train size:", len(self.train_data))
        print("Val size:", len(self.val_data))
        print("Test size:", len(self.test_data))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)