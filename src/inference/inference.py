import os
import argparse
from pathlib import Path
import random

import torch
import matplotlib.pyplot as plt

from src.data.cardiac_mri_dataset import CardiacMRIDataset
from src.models.attention_unet import get_model
from logs.logger import get_logger
from constants import VAL_ANNOT_DIR, VAL_IMG_DIR
from src.utils.load_yaml import get_yaml

logger = get_logger(name="inference.py", log_file="logs/inference.log")
logger.info("Inference pipeline started")

def parse_args(cfg):
    """Parse command-line arguments and fallback to config if not provided."""
    parser = argparse.ArgumentParser(description="Inference pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default=cfg["model"]["name"],
        help="Model to use. See model registry for available names"
    )
    parser.add_argument(
        "--inference_number",
        type=int,
        default=10,
        help="Number of patients to infer"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="figures",
        help="Directory to save inference images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="registry",
        help="Direcotry containing models"
    )
    return parser.parse_args()


def load_model(model_name: str, registry_dir: str, device: torch.device):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_name (str): Checkpoint filename (e.g., "attention_unet_best.pt")
        registry_dir (str): Path to the folder containing checkpoint files
        device (torch.device): Device to load the model onto (CPU or GPU)

    Returns:
        torch.nn.Module: Model loaded with trained weights, ready for inference
    """
    registry_dir = Path(registry_dir)
    checkpoint_path = registry_dir / f"{model_name}.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading model checkpoint: {checkpoint_path}")

    # Initialize architecture (remove file extension for registry lookup)
    model_key = checkpoint_path.stem  # removes .pt
    model = get_model(model_key).to(device)
    model.eval()

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # fallback if state_dict was saved directly
        model.load_state_dict(checkpoint)

    logger.info(f"Model '{model_key}' loaded successfully")
    return model

def main():

    try:
        cfg = get_yaml()
        args = parse_args(cfg)

        model_name = args.model
        num_patients = args.inference_number
        save_dir = os.path.join(Path(__file__).resolve().parents[2],args.save_dir)
        registry_dir = os.path.join(Path(__file__).resolve().parents[2],args.model_path)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(registry_dir,exist_ok=True)

        logger.info(f"Using model: {model_name}")
        logger.info(f"Number of patients to visualize: {num_patients}")
        logger.info(f"Inference results will be saved to: {save_dir}")

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_name,registry_dir,device)

        # Load validation dataset
        dataset_val = CardiacMRIDataset(VAL_IMG_DIR, VAL_ANNOT_DIR)
        random_indices = random.sample(range(len(dataset_val)), min(num_patients, len(dataset_val)))

        logger.info(f"Randomly selected patient indices: {random_indices}")

        # Run inference
        with torch.no_grad():
            for i, idx in enumerate(random_indices):
                image, mask = dataset_val[idx]
                image = image.unsqueeze(0).to(device, dtype=torch.float32)
                mask_2d = mask.squeeze().to(device, dtype=torch.float32)

                # Forward pass
                output = model(image)
                pred = torch.sigmoid(output)[0, 0]
                pred_mask = (pred > 0.5).float()

                # Ensure black background
                if pred_mask.mean() > 0.5:
                    pred_mask = 1 - pred_mask

                # Plotting
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(image[0, 0].cpu(), cmap='gray')
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                axes[1].imshow(image[0, 0].cpu(), cmap='gray')
                axes[1].imshow(pred_mask.cpu(), cmap='Blues', alpha=0.5)
                axes[1].contour(mask_2d.cpu(), colors='red', linewidths=1)
                axes[1].set_title("Prediction Overlay")
                axes[1].axis('off')

                plt.suptitle(f"Patient {idx}", fontsize=16)
                plt.tight_layout()

                # Save figure
                fig_path = os.path.join(save_dir, f"patient_{idx}.png")
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close(fig)  # free memory

                logger.info(f"Saved inference figure for patient {idx} at {fig_path}")

        logger.info("Inference pipeline completed successfully")

    except Exception as e:
        logger.exception("Inference pipeline failed")
        raise


if __name__ == "__main__":
    main()
