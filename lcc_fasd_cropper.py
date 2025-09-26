import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

class Crop_LCCFASD:
    def __init__(self, padding: float = 0.15):
        """
        Args:
            padding: fraction of width/height to expand bounding box
        """
        self.padding = padding

    def read_bbox_txt(self, txt_path):
        """Read bbox coordinates from a txt file. Expected format: [x1, y1, x2, y2, confidence]"""
        with open(txt_path, "r") as f:
            line = f.readline().strip()
        coords = line.strip("[]").split(",")
        coords = [float(c) for c in coords]
        x1, y1, x2, y2, conf = coords
        return x1, y1, x2, y2, conf

    def crop_and_save(self, rgb_path, depth_path, txt_path, save_base_dir, relative_path):
        # Load images
        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb_img is None or depth_img is None:
            print(f"Could not read image: {rgb_path} or {depth_path}")
            return

        # Read bbox
        x1, y1, x2, y2, conf = self.read_bbox_txt(txt_path)

        # Add padding
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w * self.padding))
        y1 = max(0, int(y1 - h * self.padding))
        x2 = min(rgb_img.shape[1], int(x2 + w * self.padding))
        y2 = min(rgb_img.shape[0], int(y2 + h * self.padding))

        # Crop
        rgb_crop = rgb_img[y1:y2, x1:x2]
        depth_crop = depth_img[y1:y2, x1:x2]

        # Save
        folder = os.path.join(save_base_dir, relative_path)
        os.makedirs(folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(rgb_path))[0]
        rgb_save_path = os.path.join(folder, f"{base_name}.jpg")
        depth_save_path = os.path.join(folder, f"{base_name}_depth.jpg")

        cv2.imwrite(rgb_save_path, rgb_crop)
        cv2.imwrite(depth_save_path, depth_crop)

    def process_dataset(self, rgb_base_dir, depth_base_dir, txt_base_dir, save_base_dir):
        subfolders = ["LCC_FASD_training", "LCC_FASD_development", "LCC_FASD_evaluation"]
        for subfolder in subfolders:
            rgb_folder = os.path.join(rgb_base_dir, subfolder)
            
            # loop over real/spoof
            for label in ["real", "spoof"]:
                rgb_label_folder = os.path.join(rgb_folder, label)
                if not os.path.exists(rgb_label_folder):
                    continue

                all_files = [f for f in os.listdir(rgb_label_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

                for img_name in tqdm(all_files, desc=f"Processing {subfolder}/{label}"):
                    rgb_path = os.path.join(rgb_label_folder, img_name)
                    
                    # Depth and TXT: directly under subfolder (no label)
                    depth_name = os.path.splitext(img_name)[0] + "_depth.jpg"
                    depth_path = os.path.join(depth_base_dir, subfolder, depth_name)
                    txt_name = os.path.splitext(img_name)[0] + "_info.txt"
                    txt_path = os.path.join(txt_base_dir, subfolder, txt_name)

                    # Save path preserves label (real/spoof)
                    relative_path = os.path.join(subfolder, label)
                    self.crop_and_save(rgb_path, depth_path, txt_path, save_base_dir, relative_path)


if __name__ == "__main__":
    rgb_base_dir = "LCC_FASD" 
    depth_base_dir = txt_base_dir ="LCC_FASD_depth" 
    #  = "LCC_FASD_depth" 
    save_base_dir = "cropped_LCC_FASD"

    cropper = Crop_LCCFASD(padding=0.15)
    cropper.process_dataset(rgb_base_dir, depth_base_dir, txt_base_dir, save_base_dir)
