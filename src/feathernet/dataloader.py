import pytorch_lightning as pl
import os 
import cv2
import albumentations as A

from torch.utils.data import DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

from dataset import CelebASpoofDataset  


class CelebASpoofDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, label_dir, batch_size=64, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_transform = A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.ISONoise(
                color_shift=(0.15,0.35), 
                intensity=(0.1,0.5), 
                p=0.05),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2,0.2),
                contrast_limit=(-0.2,0.2),
                brightness_by_max=True,
                p=0.125),
            A.MotionBlur(
                blur_limit=3, 
                p=0.2),
            A.CoarseDropout(
                num_holes_range=(3, 6),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill="random_uniform",
                p=1.0
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.5931, 0.4690, 0.4229],
                            std=[0.2471, 0.2214, 0.2157]),
            ToTensorV2(),
        ])

        # Transformation for the input test/val images
        test_transform = A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.Normalize(
                mean=[0.5931, 0.4690, 0.4229],
                std=[0.2471, 0.2214, 0.2157]
            ),
            ToTensorV2(),
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