from src.utils.load_yaml import get_yaml
from constants import (
    PREPROCESSED_CONTOURS_PATH,
    PREPROCESSED_IMAGES_PATH,
    SPLIT_DIR
)
from src.data.ingest import load_and_organize_data
from src.data.preprocessing import preprocessing_dataset

import os
import shutil
import random
import numpy as np
from pathlib import Path
from logs.logger import get_logger

# Initialize logger
logger = get_logger(name="split.py")
cfg = get_yaml()

def split_dataset():
    """
    Split preprocessed images and contours into training and validation sets.

    Args:
        cfg_path (str, optional): Path to YAML config. If None, uses default get_yaml().
    """
    try:

        # Load configi
        seed = cfg["seed"]
        split_ratio = cfg["split_ratio"]  

        random.seed(seed)

        # Define directories
        TRAIN_IMAGE_DIR = os.path.join(SPLIT_DIR, "train")
        TRAIN_ANNOT_DIR = os.path.join(SPLIT_DIR, "trainannot")
        VAL_IMG_DIR = os.path.join(SPLIT_DIR, "val")
        VAL_ANNOT_DIR = os.path.join(SPLIT_DIR, "valannot")

        # Create directories if they don't exist
        for d in [TRAIN_IMAGE_DIR, TRAIN_ANNOT_DIR, VAL_IMG_DIR, VAL_ANNOT_DIR]:
            os.makedirs(d, exist_ok=True)

        # Gather all paired files
        image_files = [
            f for f in os.listdir(PREPROCESSED_IMAGES_PATH)
            if f.endswith(".npy") and os.path.exists(os.path.join(PREPROCESSED_CONTOURS_PATH, f))
        ]

        if not image_files:
            logger.warning("No paired image/contour files found.")
            return

        # shuffle deterministically
        random.shuffle(image_files)

        # Split
        split_index = int(split_ratio * len(image_files))
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        logger.info(f"Total paired slices: {len(image_files)}")
        logger.info(f"Training slices: {len(train_files)}")
        logger.info(f"Validation slices: {len(val_files)}")

        def copy_pair(files, src_img_dir, src_annot_dir, dst_img_dir, dst_annot_dir):
            for filename in files:
                image_path = os.path.join(src_img_dir, filename)
                contour_path = os.path.join(src_annot_dir, filename)

                # Check existence
                if not os.path.exists(image_path):
                    logger.error(f"Missing image: {image_path}")
                    continue
                if not os.path.exists(contour_path):
                    logger.error(f"Missing contour: {contour_path}")
                    continue

                # Check shapes
                image_array = np.load(image_path)
                contour_array = np.load(contour_path)
                if image_array.shape != contour_array.shape:
                    logger.error(f"Shape mismatch in {filename}: {image_array.shape} vs {contour_array.shape}")
                    continue

                # Copy files
                try:
                    shutil.copy2(image_path, os.path.join(dst_img_dir, filename))
                    shutil.copy2(contour_path, os.path.join(dst_annot_dir, filename))
                    logger.info(f"Copied {filename} to train/val folders")
                except Exception as e:
                    logger.exception(f"Failed to copy {filename}: {e}")

        # Copy files
        copy_pair(train_files, PREPROCESSED_IMAGES_PATH, PREPROCESSED_CONTOURS_PATH, TRAIN_IMAGE_DIR, TRAIN_ANNOT_DIR)
        copy_pair(val_files, PREPROCESSED_IMAGES_PATH, PREPROCESSED_CONTOURS_PATH, VAL_IMG_DIR, VAL_ANNOT_DIR)

        logger.info("Data split completed successfully.")
    except Exception as e:
        logger.exception("Error while splitting dataset")


if __name__ == "__main__":
    load_and_organize_data()
    preprocessing_dataset()
    split_dataset()
