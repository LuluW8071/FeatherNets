import os
import numpy as np
import pytorch_lightning as pl
import albumentations as A
import cv2
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import torch

np.random.seed(1234)


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
        if depth.ndim == 2:  # [H,W]
            depth = np.expand_dims(depth, -1)  # [H,W,1]
        elif depth.ndim == 3 and depth.shape[2] == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            depth = np.expand_dims(depth, -1)
        depth = depth.astype(np.float32)
        depth = depth / 255.0

        result = self.alb_transform(image=depth)
        out = result["image"]  # [1,H,W]
        return out

# -------------------------------
# LCC-FASD Dataset
# -------------------------------
class LCCFASDDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None, map_transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.map_transform = map_transform
        self.samples = []

        for label_name, bin_label in [("real", 0), ("spoof", 1)]:
            folder = os.path.join(base_dir, f"{split}/{label_name}")
            if not os.path.exists(folder):
                continue
            for file in os.listdir(folder):
                if file.lower().endswith(".jpg") and not file.endswith("_depth.jpg"):
                    rgb_path = os.path.join(folder, file)
                    depth_path = os.path.join(
                        folder, os.path.splitext(file)[0] + "_depth.jpg"
                    )
                    self.samples.append((rgb_path, depth_path, bin_label))

        np.random.shuffle(self.samples)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, bin_label = self.samples[idx]

        # --- RGB ---
        img = cv2.imread(rgb_path)
        if img is None:
            raise ValueError(f"Failed to load RGB image: {rgb_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Depth ---
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to load depth image: {depth_path}")

        # Apply transforms
        if self.transform:
            img = self.transform(img)
        if self.map_transform:
            depth = self.map_transform(depth)

        # Label: use float32 if BCE, int64 if CrossEntropy
        bin_label = torch.tensor(bin_label, dtype=torch.float32)

        return img, (bin_label, depth)

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
            A.ISONoise(color_shift=(0.15,0.35), intensity=(0.1,0.5), p=0.05),
            A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2,0.2), p=0.125),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.CoarseDropout(num_holes_range=(3,6),
                            hole_height_range=(10,20),
                            hole_width_range=(10,20),
                            fill_value=0, p=0.125),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))

        test_transform = AlbumentationsTransform(A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
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
