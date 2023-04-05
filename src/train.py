import argparse
import copy
import os
import albumentations as A
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SkySegmentationDataset
from model import SegmentationModel
from prepare_ade20k_dataset import (DEFAULT_TRAIN_IMAGE_FOLDER,
                                    DEFAULT_TRAIN_MASK_FOLDER,
                                    DEFAULT_VAL_IMAGE_FOLDER,
                                    DEFAULT_VAL_MASK_FOLDER)
from segmentation_metrics import dice, iou

# Default folder used to save the trained model
DEFAULT_SAVE_FOLDER: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"../results")

# Default model initialization parameters
DEFAULT_MODEL_NAME: str = "deeplabv3_mobilenetv3"
DEFAULT_NUM_CLASSES: int = 1
DEFAULT_PRETRAINED_WEIGHTS: str = "DEFAULT"
# Normalization parameters for backbone
NORMALIZE_MEAN: list[float] = [0.485, 0.456, 0.406]
NORMALIZE_STD: list[float] = [0.229, 0.224, 0.225]
# Default resize parameters
DEFAULT_IMG_WIDTH: int = 520
DEFAULT_IMG_HEIGHT: int = 520
# Default training parameters
DEFAULT_NUM_EPOCHS: int = 5
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_MAX_LEARNING_RATE: float = 1e-3


def train_fn(
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler,
        device: torch.device = torch.device("cpu")
    ) -> float:
    """
    Train the given model for one epoch using the given DataLoader.

    Args:
        loader (torch.utils.data.DataLoader): A DataLoader that contains the training dataset.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        device (torch.device, optional): The device to run the model on. Default is CPU.

    Returns:
        float: The running loss for the epoch.
    """
    loop = tqdm(loader)
    model.train()
    train_loss = 0
    for imgs, masks in loop:
        imgs = imgs.to(device=device)
        masks = masks.float().unsqueeze(1).to(device=device)

        # forward
        predictions = model(imgs)["out"]
        loss = criterion(predictions, masks)

        # backward
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * imgs.size(0)
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    train_loss /= len(loader.sampler)

    return train_loss

def evaluate_fn(
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        device: torch.device = torch.device("cpu")
    ) -> tuple[float, float, float]:
    """
    Evaluate the given model on the given DataLoader.

    Args:
        loader (torch.utils.data.DataLoader): A DataLoader that contains the validation dataset.
        model (torch.nn.Module): The model to be evaluated.
        criterion (torch.nn.Module): The loss function used for evaluation.
        device (torch.device, optional): The device to run the model on. Default is CPU.

    Returns:
        tuple[float, float, float]: A tuple containing the validation loss, IoU score, and Dice score.
    """
    model.eval()
    iou_score = 0
    dice_score = 0
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device).unsqueeze(1)
            predictions = model(imgs)["out"]
            loss = criterion(predictions, masks)
            predictions = torch.sigmoid(predictions)
            predictions = (predictions > 0.5).float()
            # update running validation loss
            val_loss += loss.item() * imgs.size(0)
            iou_score += iou(predictions.cpu().numpy(), masks.cpu().numpy()) * imgs.size(0)
            dice_score += dice(predictions.cpu().numpy(), masks.cpu().numpy()) * imgs.size(0)

        val_loss /= len(loader.sampler)
        iou_score /= len(loader.sampler)
        dice_score /= len(loader.sampler)

    return val_loss, iou_score, dice_score

def main():
    """
    Parse command line arguments, initialize the model, parameters, and hyperparameters, train the model,
    and save the best model.
    """
    args = parse_args()

    config_file = args.config_file
    if config_file is None or os.path.splitext(config_file)[-1].lower() != ".yaml" or not os.path.exists(config_file):
        # TODO: switch to logger
        print(f"{config_file} is not a valid yaml file, default parameters will be used.")
        config = {}
    else:
        with open(config_file, "r", encoding="utf-8") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Redundant but if the script is called with --train_image_folder empty the default value is not set
    train_image_folder = (args.train_image_folder if args.train_image_folder
        else DEFAULT_TRAIN_IMAGE_FOLDER)
    train_mask_folder = (args.train_mask_folder if args.train_mask_folder
        else DEFAULT_TRAIN_MASK_FOLDER)
    val_image_folder = (args.val_image_folder if args.val_image_folder
        else DEFAULT_VAL_IMAGE_FOLDER)
    val_mask_folder = (args.val_mask_folder if args.val_mask_folder
        else DEFAULT_VAL_MASK_FOLDER)

    image_height = config.get("image_height", DEFAULT_IMG_HEIGHT)
    image_width = config.get("image_width", DEFAULT_IMG_HEIGHT)
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=30, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=NORMALIZE_MEAN,
                std=NORMALIZE_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=NORMALIZE_MEAN,
                std=NORMALIZE_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)

    train_data = SkySegmentationDataset(
        image_dir=train_image_folder,
        mask_dir=train_mask_folder,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    val_data = SkySegmentationDataset(
        image_dir=val_image_folder,
        mask_dir=val_mask_folder,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )

    # Init the segmentation model

    model_name = config.get("model_name", DEFAULT_MODEL_NAME)
    num_classes = config.get("num_classes", DEFAULT_NUM_CLASSES)
    pretrained_weights = config.get("pretrained_weights", DEFAULT_PRETRAINED_WEIGHTS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segmentation_model = SegmentationModel(
        model_name=model_name,
        pretrained_weights=pretrained_weights,
        num_classes=num_classes,
        device=device,
    )

    model = segmentation_model.model
    # Set optimizer, scheduler and loss with config
    learning_rate = config.get("learning_rate", DEFAULT_LEARNING_RATE)
    max_learning_rate = config.get("max_learning_rate", DEFAULT_MAX_LEARNING_RATE)
    num_epochs = config.get("num_epochs", DEFAULT_NUM_EPOCHS)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_learning_rate,
        total_steps=num_epochs * len(train_loader)
    )
    criterion = nn.BCEWithLogitsLoss()

    min_val_loss = np.Inf
    best_segm_model = segmentation_model
    mlflow.set_experiment(f"{model_name}_lr{learning_rate}_maxlr_{DEFAULT_MAX_LEARNING_RATE}_{num_epochs}")
    mlflow.start_run()
    for epoch in range(num_epochs):
        train_loss = train_fn(train_loader, model, optimizer, criterion, scheduler, device=device)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        val_loss, iou_score, dice_score = evaluate_fn(val_loader, model, criterion, device=device)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("iou_score", iou_score, step=epoch)
        mlflow.log_metric("dice_score", dice_score, step=epoch)
        if val_loss < min_val_loss:
            best_segm_model = copy.deepcopy(segmentation_model)
            min_val_loss = val_loss

    save_folder = (args.save_folder if args.save_folder
                   else DEFAULT_SAVE_FOLDER)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    model_path = os.path.join(save_folder, f"{model_name}_lr{learning_rate}_maxlr_{DEFAULT_MAX_LEARNING_RATE}_{num_epochs}.pth")
    best_segm_model.save(model_path)
    mlflow.log_artifact(model_path)
    mlflow.end_run()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_folder", type=str, default=DEFAULT_SAVE_FOLDER,
                        help="Folder where to save the trained model.")

    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to the yaml config file.")

    parser.add_argument("--train_image_folder", type=str, default=DEFAULT_TRAIN_IMAGE_FOLDER,
                        help="Path to the folder where the training images are stored.",
                        nargs='?')

    parser.add_argument("--train_mask_folder", type=str, default=DEFAULT_TRAIN_MASK_FOLDER,
                        help="Path to the folder where the training masks are stored.",
                        nargs='?')

    parser.add_argument("--val_image_folder", type=str, default=DEFAULT_VAL_IMAGE_FOLDER,
                        help="Path to the folder where the validation images are stored.",
                        nargs='?')

    parser.add_argument("--val_mask_folder", type=str, default=DEFAULT_VAL_MASK_FOLDER,
                        help="Path to the folder where the validation masks are stored.",
                        nargs='?')

    return parser.parse_args()

if __name__ == "__main__":
    main()
