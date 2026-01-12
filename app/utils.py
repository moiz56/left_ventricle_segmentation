
import os
import matplotlib.pyplot as plt

from logs.logger import get_logger


logger = get_logger(name="api", log_file="logs/api.log")

def save_slice_overlays(predicted_slices, original_slices, save_dir):
    """Save slice overlays and predicted masks in the background."""
    try:
        for idx, (mask, orig) in enumerate(zip(predicted_slices, original_slices)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Original image
            axes[0].imshow(orig, cmap='gray')
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            # Overlay predicted mask
            axes[1].imshow(orig, cmap='gray')
            axes[1].imshow(mask, cmap='Blues', alpha=0.5)
            axes[1].set_title("Prediction Overlay")
            axes[1].axis('off')

            plt.suptitle(f"Slice {idx}", fontsize=16)
            plt.tight_layout()

            fig_path = os.path.join(save_dir, f"slice_{idx}.png")
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close(fig)

            logger.info(f"Saved overlay {fig_path}")

    except Exception as e:
        logger.exception(f"Background save failed: {e}")