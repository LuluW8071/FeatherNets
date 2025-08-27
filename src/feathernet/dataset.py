import os
import numpy as np
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset

class CelebASpoofDataset(Dataset):
    """
    Dataset for CelebA-Spoof.

    Returns:
        image: Tensor (after Albumentations ToTensorV2)
        label: int (0=spoof, 1=live)
    """
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # List of tuples (img_path, label, bbox)

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

        # Load image
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)  # HWC

        # Crop face safely
        cropped_np = self._crop_face(image_np, bbox)

        # Handle empty crops
        if cropped_np.size == 0 or cropped_np.shape[0] == 0 or cropped_np.shape[1] == 0:
            # fallback: use full image
            cropped_np = image_np

        # Apply transform
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=cropped_np)
                cropped_np = augmented['image']
            else:  # torchvision transforms
                cropped_np = self.transform(Image.fromarray(cropped_np))

        return cropped_np, label

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
        h, w = img_np.shape[:2]

        # Check bbox validity
        if not bbox or len(bbox) < 4:
            # fallback to full image
            return img_np

        try:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, x1 + max(1, int(bbox[2])))  # width >= 1
            y2 = min(h, y1 + max(1, int(bbox[3])))  # height >= 1

            cropped = img_np[y1:y2, x1:x2, :]
            # fallback if crop is empty
            if cropped.size == 0:
                return img_np
            return cropped

        except Exception as e:
            # any unexpected error, return full image
            print(f"Skipping crop due to error: {e}")
            return img_np