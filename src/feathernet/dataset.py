import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class CelebASpoofDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        """
        Args:
            data_dir (str): Base directory containing image files.
            label_file (str): Path to the label file.
            transform: PyTorch transform to apply on the cropped face image.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # (image_path, label, bbox)

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_rel, label = parts[0], int(parts[1])
                img_path = os.path.join(data_dir, img_rel.split("/", 1)[-1])
                bb_path = img_path[:-4] + "_BB.txt"

                if not os.path.exists(img_path) or not os.path.exists(bb_path):
                    continue

                try:
                    bbox = self._read_bbox(bb_path)
                    self.samples.append((img_path, 1 if label == 1 else 0, bbox))
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, bbox = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        cropped_np = self._crop_face(image_np, bbox)
        cropped = Image.fromarray(cropped_np)

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, label

    def _read_bbox(self, bb_path):
        """Read bounding box from _BB.txt file."""
        with open(bb_path, "r") as f:
            line = f.readline()
        bbox = [int(x) for x in line.strip().split()[:4]]  # x, y, w, h
        return bbox

    @staticmethod
    def _clamp(x, min_x, max_x):
        return min(max(x, min_x), max_x)

    def _crop_face(self, img_np, bbox):
        """Crop face from numpy array using bounding box, safely clamped."""
        if len(img_np.shape) == 2:
            real_h, real_w = img_np.shape
        else:
            real_h, real_w, _ = img_np.shape

        x1 = self._clamp(int(bbox[0] * (real_w / 224)), 0, real_w)
        y1 = self._clamp(int(bbox[1] * (real_h / 224)), 0, real_h)
        w1 = int(bbox[2] * (real_w / 224))
        h1 = int(bbox[3] * (real_h / 224))

        if len(img_np.shape) == 2:
            cropped_face = img_np[y1:self._clamp(y1 + h1, 0, real_h),
                                  x1:self._clamp(x1 + w1, 0, real_w)]
        else:
            cropped_face = img_np[y1:self._clamp(y1 + h1, 0, real_h),
                                  x1:self._clamp(x1 + w1, 0, real_w), :]
        return cropped_face
