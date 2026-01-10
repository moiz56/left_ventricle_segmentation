import os
from pathlib import Path


# dataset path constants
BASE_DIR = Path(__file__).resolve().parent
DATASET_BASE_DIR = os.path.join(BASE_DIR,"dataset","emidec-dataset-1.0.1")

#organizing data constants
ORGANIZED_DATASET_PATH = os.path.join(BASE_DIR,"dataset","reorganized_dataset")

# preprocessing dataset constants
PREPROCESSED_DATASET_DIR = os.path.join(BASE_DIR,"dataset","emidec_preprocessed")
IMAGES_OUTPUT_PATH = os.path.join(ORGANIZED_DATASET_PATH, "images")
CONTOURS_OUTPUT_PATH = os.path.join(ORGANIZED_DATASET_PATH, "contours")
PREPROCESSED_IMAGES_PATH = os.path.join(PREPROCESSED_DATASET_DIR,"images")
PREPROCESSED_CONTOURS_PATH = os.path.join(PREPROCESSED_DATASET_DIR,"contours")

# splitting image constants
SPLIT_DIR = os.path.join(BASE_DIR,"dataset","split_emidec")
TRAIN_IMAGE_DIR = os.path.join(SPLIT_DIR, "train")
TRAIN_ANNOT_DIR= os.path.join(SPLIT_DIR, "trainannot")
VAL_IMG_DIR = os.path.join(SPLIT_DIR, "val")
VAL_ANNOT_DIR = os.path.join(SPLIT_DIR, "valannot")
