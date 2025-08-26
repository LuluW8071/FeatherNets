import pytorch_lightning as pl
import os 
import cv2
import albumentations as A

from torch.utils.data import DataLoader
from torchvision import transforms
from albumentations.augmentations.blur import MotionBlur
from albumentations.pytorch import ToTensorV2

from dataset import CelebASpoofDataset  


class CelebASpoofDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, label_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 1  # Binary classification: live vs spoof
        self.pos_weight = None

    def setup(self, stage=None):
        # Transforms
        train_transform = A.Compose([
            A.Resize(224, 224,interpolation=cv2.INTER_CUBIC),
            A.augmentations.transforms.ISONoise(
                color_shift=(0.15,0.35), 
                intensity=(0.1,0.5), 
                always_apply=False, 
                p=0.05),
            A.augmentations.transforms.RandomBrightnessContrast(
                brightness_limit=(-0.2,0.2),
                contrast_limit=(-0.2,0.2),
                always_apply=False,
                brightness_by_max=True,
                p=0.125),
            MotionBlur(
                blur_limit=3, 
                p=0.2),
            A.augmentations.transforms.ImageCompression(
                quality_lower=50,
                quality_upper=100,
                always_apply=False,
                p=0.25),
            # Cutout
            A.augmentations.dropout.CoarseDropout(
                max_holes=24,
                max_height=8,
                max_width=8,
                min_holes=4,
                min_height=4,
                min_width=4,
                fill_value=0,
                always_apply=False,
                p=0.25),
            # Pixel dropout
            A.augmentations.transforms.GaussNoise(
                var_limit=(10.0, 50.0),
                mean=0,
                always_apply=False,
                p=0.2),
            # To tensor
            A.Normalize(mean=[0.5931, 0.4690, 0.4229],
                            std=[0.2471, 0.2214, 0.2157]),
            ToTensorV2(),
            # Convert from [0, 255] to [0.0, 1.0]


        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])

        # Datasets
        self.train_data = CelebASpoofDataset(
            data_dir=self.data_dir,
            label_file=os.path.join(self.label_dir, "train_label.txt"),
            transform=train_transform
        )
        self.val_data = CelebASpoofDataset(
            data_dir=self.data_dir,
            label_file=os.path.join(self.label_dir, "test_label.txt"),
            transform=test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


if __name__ == "__main__":

    # Paths
    data_dir = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/versions/2/CelebA_Spoof_/CelebA_Spoof/Data"
    label_dir = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing/versions/2/CelebA_Spoof_/CelebA_Spoof/metas/intra_test"

    train_dataset = CelebASpoofDataset(data_dir, os.path.join(label_dir, "train_label.txt"))
    test_dataset  = CelebASpoofDataset(data_dir, os.path.join(label_dir, "test_label.txt"))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # images, labels = next(iter(train_loader))
    # print("Batch shape:", images.shape)
    # print("Labels (0=spoof, 1=live):", labels[:10])