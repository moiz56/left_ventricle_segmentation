import os
import argparse
from pathlib import Path

import mlflow

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.data.ingest import load_and_organize_data
from src.data.preprocessing import preprocessing_dataset
from src.data.remap_masks import remap_masks
from src.data.cardiac_mri_dataset import prepare_data
from src.models.attention_unet import get_model
from src.training.train import train_model

from logs.logger import get_logger
from constants import TRAIN_ANNOT_DIR, TRAIN_IMAGE_DIR, VAL_ANNOT_DIR, VAL_IMG_DIR, SPLIT_DIR
from src.utils.load_yaml import get_yaml

def parse_args(cfg):
    """Parse command-line arguments and fallback to config if not provided."""
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument("--lr", type=float, default=float(cfg.get("lr")), help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=cfg.get("batch_size"), help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=cfg.get("num_epochs"), help="Number of epochs")
    parser.add_argument("--model", type=str, default=cfg["model"]["name"], help="Model to train")
    args = parser.parse_args()
    return args


def plot_training_history(history, save_dir="figures"):
    save_dir = Path(__file__).resolve().parents[2] / save_dir
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(10,4))
    plt.plot(epochs, history["train_loss"], 'b-', label="Train Loss")
    plt.plot(epochs, history["val_loss"], 'r-', label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = save_dir / "loss_plot.png"
    plt.savefig(loss_plot_path, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(loss_plot_path, artifact_path="plots")  # log after saving

    # Dice
    plt.figure(figsize=(10,4))
    plt.plot(epochs, history["val_dice"], 'g-', label="Val Dice")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Score")
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)
    dice_plot_path = save_dir / "dice_plot.png"
    plt.savefig(dice_plot_path, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(dice_plot_path, artifact_path="plots")  # log after saving

    print(f"Training history plots saved to {save_dir}")

def main():
    logger = get_logger(name="training_pipeline.py", log_file="logs/training_pipeline.log")
    logger.info("Training pipeline started")

    try:
        mlflow.set_experiment("attention_unet")
        with mlflow.start_run(run_name="training_run"):
            # Load config
            cfg = get_yaml()

            # Parse CLI arguments
            args = parse_args(cfg)
            lr = args.lr
            batch_size = args.batch_size
            num_epochs = args.num_epochs
            model = args.model

            logger.info(f"Using hyperparameters: lr={lr}, batch_size={batch_size}, num_epochs={num_epochs}")

            # Create registry folder
            registry_dir = Path(__file__).resolve().parents[2] / "registry"
            registry_dir.mkdir(parents=True, exist_ok=True)
            save_path = registry_dir / cfg.get("model_save_path", "model.pt")
            logger.info(f"Model will be saved to: {save_path}")

            # Data pipeline
            logger.info("Loading and organizing dataset")
            load_and_organize_data()
            logger.info("Dataset loaded and organized")

            logger.info("Preprocessing dataset")
            preprocessing_dataset()
            logger.info("Dataset preprocessed")

            logger.info("Remapping masks")
            remap_masks(base_dir=SPLIT_DIR)
            logger.info("Masks remapped")

            # Prepare dataloaders
            logger.info("Preparing dataset")
            train_loader, val_loader = prepare_data(
                TRAIN_IMAGE_DIR, TRAIN_ANNOT_DIR, VAL_IMG_DIR, VAL_ANNOT_DIR, batch_size
            )
            imgs, masks = next(iter(train_loader))
            logger.info("Train batch shapes: images=%s, masks=%s", imgs.shape, masks.shape)
            logger.info("Dataset prepared")

            # Model + optimizer
            logger.info("Preparing model")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mlflow.log_param("model.name",model)
            model = get_model(model)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            logger.info("Model initialized on device: %s", device)

            # Training
            logger.info("Starting training")
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size",batch_size)
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                device=device,
                num_epochs=num_epochs,
                save_path=save_path
            )
            logger.info("Training complete")

            plot_training_history(history)

    except Exception as e:
            logger.exception("Training pipeline failed: %s", e)
            raise

if __name__ == "__main__":
    main()
