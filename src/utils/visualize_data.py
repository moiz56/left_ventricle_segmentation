import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

def visualize(img_path="dataset/emidec-dataset-1.0.1/Case_N006/Images/Case_N006.nii.gz",
              mask_path="dataset/emidec-dataset-1.0.1/Case_N006/Contours/Case_N006.nii.gz",
              slice_idx=1,
              time_idx=0):
    try:
        # Load NIfTI files
        try:
            img = nib.load(img_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        try:
            msk = nib.load(mask_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        img_data = img.get_fdata()
        mask_data = msk.get_fdata()

        # Validate slice index
        if slice_idx >= img_data.shape[2]:
            raise IndexError(f"slice_idx {slice_idx} is out of bounds for image with {img_data.shape[2]} slices")

        # Extract 2D slice
        img_slice = img_data[:, :, slice_idx, time_idx] if img_data.ndim == 4 else img_data[:, :, slice_idx]
        mask_slice = mask_data[:, :, slice_idx, time_idx] if mask_data.ndim == 4 else mask_data[:, :, slice_idx]
        mask_bin = mask_slice > 0

        # Prepare figure folder
        root = Path(__file__).resolve().parents[2]
        fig_dir = root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Plot image and mask
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img_slice, cmap="gray")
        ax1.set_title(f"Image slice {slice_idx}")
        ax1.axis("off")

        ax2.imshow(img_slice, cmap="gray")
        ax2.imshow(mask_bin, cmap="Reds", alpha=0.5)
        ax2.set_title(f"Image + Mask (slice {slice_idx})")
        ax2.axis("off")

        plt.tight_layout()
        fig_path = fig_dir / f"Case_N006_slice_{slice_idx}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"Saved figure to {fig_path}")

    except Exception as e:
        raise ValueError(f"Error generating image and/or mask: {e}")


