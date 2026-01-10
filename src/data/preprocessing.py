import os
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.transform import resize
from logs.logger import get_logger
from src.data.ingest import load_and_organize_data

from constants import IMAGES_OUTPUT_PATH,CONTOURS_OUTPUT_PATH,PREPROCESSED_DATASET_DIR

logging = get_logger(name="preprocessing.py")

def pad_or_resize_to_224(slice_2d, is_mask=False):
    """
    Pad a 2D slice with zeros if smaller than 224x224.
    Resize down if larger than 224x224.

    Args:
        slice_2d (np.ndarray): 2D image or mask slice
        is_mask (bool): True if processing a segmentation mask (nearest interpolation)
    """

    h, w = slice_2d.shape

    # If larger than 224 in either dimension → resize down
    if h > 224 or w > 224:
        order = 0 if is_mask else 1  # 0=nearest, 1=bilinear
        resized = resize(
            slice_2d,
            (224, 224),
            order=order,
            preserve_range=True,
            anti_aliasing=not is_mask
        ).astype(slice_2d.dtype)
        return resized

    # Otherwise → pad up to 224
    pad_h = 224 - h
    pad_w = 224 - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(
        slice_2d,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )
    return padded


def process_nii_file(nii_path, output_folder, prefix, is_mask=False):
    """Load, canonicalize, pad/resize, and save slices of a .nii file."""
    try:
        nii = nib.load(nii_path)
        nii = nib.as_closest_canonical(nii)  # Ensure RAS+ orientation
        data = nii.get_fdata()

        if data.ndim != 3:
            print(f"Skipping non-3D file: {nii_path}")
            return

        for i in range(data.shape[2]):  # Iterate over slices (assumed axial)
            slice_2d = data[:, :, i]
            processed_slice = pad_or_resize_to_224(slice_2d, is_mask=is_mask)

            # Save each slice as .npy
            filename = f"{prefix}_slice_{i:03}.npy"
            out_path = os.path.join(output_folder, filename)
            # Cast mask to int (if is_mask) to avoid floats
            if is_mask:
                np.save(out_path, processed_slice.astype(np.int64))
            else:
                np.save(out_path, processed_slice.astype(np.float32))

        logging.info(f"Processed {prefix}: {data.shape[2]} slices")

    except Exception as e:
        logging.error(f"Error processing {nii_path}: {e}")


def preprocessing_dataset():

    try:

        preprocessed_images_output_dir = os.path.join(PREPROCESSED_DATASET_DIR, "images")
        preprocessed_contours_output_dir = os.path.join(PREPROCESSED_DATASET_DIR, "contours")

        # Create output directories if they don't exist
        os.makedirs(preprocessed_images_output_dir, exist_ok=True)
        os.makedirs(preprocessed_contours_output_dir, exist_ok=True)
        # Process image files
        for nii_file in os.listdir(IMAGES_OUTPUT_PATH):
            if nii_file.endswith(".nii") or nii_file.endswith(".nii.gz"):
                full_path = os.path.join(IMAGES_OUTPUT_PATH, nii_file)
                prefix = Path(nii_file).stem
                process_nii_file(full_path, preprocessed_images_output_dir, prefix, is_mask=False)

        # Process contour files
        for nii_file in os.listdir(CONTOURS_OUTPUT_PATH):
            if nii_file.endswith(".nii") or nii_file.endswith(".nii.gz"):
                full_path = os.path.join(CONTOURS_OUTPUT_PATH, nii_file)
                prefix = Path(nii_file).stem
                process_nii_file(full_path, preprocessed_contours_output_dir, prefix, is_mask=True)

        images_len = len(os.listdir(preprocessed_images_output_dir))
        contours_len = len(os.listdir(preprocessed_images_output_dir))
        logging.info(f"Total images: {images_len}")
        logging.info(f"Total contours: {contours_len}")
        if images_len != contours_len:
            raise ValueError(
                f"Number of images ({images_len}) does not match number of contours ({contours_len})!"
            )
        else:
            logging.info("Image and contour counts match.")
    except Exception as e:
        raise RuntimeError(f"Error while preprocessing {e}")
        


if __name__ == "__main__":
    load_and_organize_data()
    preprocessing_dataset()