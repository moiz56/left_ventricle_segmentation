import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import random
from pathlib import Path
from logs.logger import get_logger

from constants import TRAIN_ANNOT_DIR, TRAIN_IMAGE_DIR, VAL_ANNOT_DIR, VAL_IMG_DIR

# Initialize logger
logger = get_logger(name="dataset.py", log_file="logs/dataset.log")

class CardiacMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False, log_samples=5):
        self.image_files = sorted(Path(image_dir).glob("*.npy"))
        self.mask_files = sorted(Path(mask_dir).glob("*.npy"))
        self.augment = augment

        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) must match"

        logger.info(f"Initialized CardiacMRIDataset: {len(self.image_files)} samples")
        logger.info(f"Augmentation enabled: {self.augment}")

        # Log first few samples to verify paths and shapes
        for i, (img_path, mask_path) in enumerate(zip(self.image_files[:log_samples], self.mask_files[:log_samples])):
            img_shape = np.load(img_path).shape
            mask_shape = np.load(mask_path).shape
            logger.info(f"Sample {i}: image={img_path.name} {img_shape}, mask={mask_path.name} {mask_shape}")
            if img_shape != mask_shape:
                logger.warning(f"Shape mismatch in sample {i}: {img_shape} vs {mask_shape}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # -------- Load and normalize --------
            img = np.load(self.image_files[idx]).astype(np.float32)  # [H, W]
            mask = np.load(self.mask_files[idx]).astype(np.uint8)    # 0 or 2

            # Remap mask labels: 0 -> 0, 2 -> 1
            mask = (mask // 2).astype(np.uint8)

            # Intensity normalization per slice
            img = np.clip(img, np.percentile(img, 0.5), np.percentile(img, 99.5))
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Convert to tensors [1, H, W]
            img  = torch.from_numpy(img).unsqueeze(0).float()
            mask = torch.from_numpy(mask).unsqueeze(0).long()

            # -------- Data Augmentations --------
            if self.augment:
                # --- random flips ---
                if random.random() > 0.5:
                    img = F.hflip(img)
                    mask = F.hflip(mask)
                if random.random() > 0.5:
                    img = F.vflip(img)
                    mask = F.vflip(mask)

                # --- random zoom with zero-padding (no interpolation) ---
                if random.random() > 0.5:
                    H, W = img.shape[1], img.shape[2]
                    scale = random.uniform(0.6, 0.9)
                    crop_h, crop_w = int(H * scale), int(W * scale)
                    top  = random.randint(0, H - crop_h)
                    left = random.randint(0, W - crop_w)

                    img_crop  = img[:, top:top + crop_h, left:left + crop_w]
                    mask_crop = mask[:, top:top + crop_h, left:left + crop_w]

                    new_img  = torch.zeros_like(img)
                    new_mask = torch.zeros_like(mask)
                    y_off = random.randint(0, H - crop_h)
                    x_off = random.randint(0, W - crop_w)
                    new_img[:, y_off:y_off + crop_h, x_off:x_off + crop_w] = img_crop
                    new_mask[:, y_off:y_off + crop_h, x_off:x_off + crop_w] = mask_crop

                    img, mask = new_img, new_mask

                # --- T1-style augmentation: contrast, sharpening, brightness ---
                if random.random() > 0.5:
                    c_factor = random.uniform(1.2, 1.5)
                    img = F.adjust_contrast(img, c_factor)

                    blurred = F.gaussian_blur(img, kernel_size=5, sigma=1.0)
                    img = img + 0.4 * (img - blurred)
                    img = img.clamp(0, 1)

                    b_factor = random.uniform(1.05, 1.15)
                    img = F.adjust_brightness(img, b_factor)

            # Log sample info for verification
            if idx < 5:  # log only first 5 for speed
                logger.info(f"Sample {idx} processed: img shape {img.shape}, mask shape {mask.shape}")

            return img, mask

        except Exception as e:
            logger.exception(f"Failed to load or process sample {idx}: {e}")
            raise

# ===== Test dataset logging =====
if __name__ == "__main__":
    try:
        dataset_train = CardiacMRIDataset(
            image_dir=TRAIN_IMAGE_DIR,
            mask_dir=TRAIN_ANNOT_DIR,
            augment=True
        )
        dataset_val = CardiacMRIDataset(
            image_dir=VAL_IMG_DIR,
            mask_dir=VAL_ANNOT_DIR,
            augment=False
        )
        logger.info("Dataset initialization completed successfully.")
    except Exception as e:
        logger.exception(f"Failed to initialize datasets: {e}")
