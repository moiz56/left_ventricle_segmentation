from constants import IMAGES_OUTPUT_PATH, CONTOURS_OUTPUT_PATH, DATASET_BASE_DIR
from logs.logger import get_logger

import os
import shutil
from pathlib import Path

logger = get_logger()

def load_and_organize_data():

    """
    This functions get the data from the dataset folder and removes the patient folder
    to reorganize it into images and their respective contours
    """

    try:
        # Ensure output directories exist
        os.makedirs(IMAGES_OUTPUT_PATH, exist_ok=True)
        os.makedirs(CONTOURS_OUTPUT_PATH, exist_ok=True)

        copied_images = 0
        copied_contours = 0

        # Walk through dataset directory
        for root, _, files in os.walk(DATASET_BASE_DIR):
            for file in files:
                if not file.endswith(".nii.gz"):
                    continue  # skip non-NIfTI files

                full_path = os.path.join(root, file)
                root_lower = root.lower()

                # Determine destination
                if "images" in root_lower:
                    dest_path = os.path.join(IMAGES_OUTPUT_PATH, file)
                    copied_images += 1
                elif "contours" in root_lower:
                    dest_path = os.path.join(CONTOURS_OUTPUT_PATH, file)
                    copied_contours += 1
                else:
                    continue  # skip if folder is neither images nor contours

                # Copy file
                try:
                    shutil.copy2(full_path, dest_path)
                    logger.info(f"Copied: {full_path} -> {dest_path}")
                except Exception as copy_err:
                    logger.warning(f"Failed to copy {full_path} -> {dest_path}: {copy_err}")

        logger.info(f"Total images copied: {copied_images}")
        logger.info(f"Total contours copied: {copied_contours}")

        # Assert equal lengths
        images_len = len(os.listdir(IMAGES_OUTPUT_PATH))
        contours_len = len(os.listdir(CONTOURS_OUTPUT_PATH))
        if images_len != contours_len:
            raise ValueError(
                f"Number of images ({images_len}) does not match number of contours ({contours_len})!"
            )
        else:
            logger.info("Image and contour counts match.")

    except Exception as e:
        raise RuntimeError(f"Error organizing dataset: {e}")

if __name__ == "__main__":
    load_and_organize_data()
