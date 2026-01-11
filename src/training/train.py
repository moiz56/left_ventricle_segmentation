import torch
from torch import optim
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from pathlib import Path
from src.utils.load_yaml import get_yaml

import os

from src.data.cardiac_mri_dataset import CardiacMRIDataset
from src.models.attention_unet import attention_unet
from logs.logger import get_logger
from constants import TRAIN_ANNOT_DIR, TRAIN_IMAGE_DIR, VAL_ANNOT_DIR, VAL_IMG_DIR, SPLIT_DIR

logger = get_logger(name="train", log_file="logs/training.log")
cfg = get_yaml()

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=50, save_path="best_model.pth"):
    """
    Trains and validates the model, saving the best checkpoint.
    """
    loss_function = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True
    )

    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean",
        get_not_nans=False
    )

    post_pred = AsDiscrete(argmax=True)
    post_label = AsDiscrete(to_onehot=2)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        dice_metric.reset()

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, dtype=torch.float32), masks.to(device, dtype=torch.long)
                outputs = model(images)
                loss = loss_function(outputs, masks)
                val_loss += loss.item()

                outputs_list = [post_pred(o) for o in decollate_batch(outputs)]
                masks_list = [post_label(m) for m in decollate_batch(masks)]
                dice_metric(y_pred=outputs_list, y=masks_list)

        val_loss /= len(val_loader)
        val_dice = dice_metric.aggregate().item()

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            try:
                torch.save(model.state_dict(), save_path)
                logger.info("Saved new best model at epoch %d with val_loss %.4f", epoch + 1, best_loss)
            except Exception as e:
                logger.exception("Failed to save model: %s", e)

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.6f | Val Loss: %.6f | Val Dice: %.4f",
            epoch + 1, num_epochs, train_loss, val_loss, val_dice
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

    logger.info("Training finished. Best validation loss: %.4f", best_loss)
    return history


def main():
    try:

        lr = float(cfg["lr"])
        batch_size = cfg["batch_size"]
        os.makedirs(os.path.join(Path(__file__).resolve().parents[2],"registry"),exist_ok=True)
        save_path = os.path.join(Path(__file__).resolve().parents[2],"registry", cfg["model_save_path"])
        num_epochs = cfg["num_epochs"]

        # Initialize datasets
        dataset_train = CardiacMRIDataset(TRAIN_IMAGE_DIR, TRAIN_ANNOT_DIR, augment=True)
        dataset_val = CardiacMRIDataset(VAL_IMG_DIR, VAL_ANNOT_DIR, augment=False)

        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        # Log batch shapes
        imgs, masks = next(iter(train_loader))
        logger.info("Train batch shapes: images=%s, masks=%s", imgs.shape, masks.shape)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = attention_unet()
        model.to(device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Start training
        history = train_model(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, save_path=save_path)

    except Exception as e:
        logger.exception("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
