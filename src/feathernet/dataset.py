import os
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset

np.random.seed(42)

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

        bin_label = torch.tensor(bin_label, dtype=torch.float32)
        return img, (bin_label, depth)
