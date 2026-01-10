from constants import SPLIT_DIR
from logs.logger import get_logger

import os
import numpy as np

logger = get_logger(name= "remap_masks.py")

def remap_masks(base_dir, splits=("trainannot", "valannot")):
    for split in splits:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            continue

        logger.info(f"Scanning {split_path} ...")

        for root, _, files in os.walk(split_path):  #walks all levels (works with or without subfolders)
            for fname in files:
                if not fname.endswith(".npy"):
                    continue

                fpath = os.path.join(root, fname)
                try:
                    mask = np.load(fpath)

                    # Convert to int
                    mask_int = mask.astype(np.int32)

                    # Remap classes (merge 1, 3, 4 into 2)
                    mask_int = np.where((mask_int == 1) | (mask_int == 3) | (mask_int == 4), 2, mask_int)

                    # Save as uint8 (overwrite)
                    np.save(fpath, mask_int.astype(np.uint8))

                    # Reload for confirmation
                    loaded = np.load(fpath)
                    logger.info(f"  {fname}: dtype={loaded.dtype}, unique labels={np.unique(loaded)}")

                except Exception as e:
                    logger.exception(f"Error processing {fpath}: {e}")

    logger.info("Finished remapping masks: classes 1, 3, and 4 merged into class 2.")

if __name__ == "__main__":
    remap_masks(SPLIT_DIR)