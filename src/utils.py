import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

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

def load_config_file(config_file: str = None) -> dict:
    """
    Loads configuration parameters from a yaml file.
    Args:
        config_file (str): Path to the yaml config file.
    Returns:
        dict: Dictionary containing the configuration parameters.
    """
    config = {}
    if (
        config_file is None
        or os.path.splitext(config_file)[-1].lower() != ".yaml"
        or not os.path.exists(config_file)
    ):
        print(f"{config_file} is not a valid yaml file, default parameters will be used.")

    else:
        with open(config_file, "r", encoding="utf-8") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return config


def display_segmentation_results(
        image: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray = None,
        suffix_segm = None
    ) -> None:
    """
    Display the input image alongside its predicted segmentation mask and ground truth mask.

    Args:
        image (np.ndarray): The input image as a numpy array.
        pred_mask (np.ndarray): The predicted binary segmentation mask as a numpy array.
        gt_mask (np.ndarray, optional): The optional ground truth binary segmentation mask.
            If not provided, only the image and predicted mask will be displayed.
    """

    if gt_mask is None:
        ncols = 2
    else:
        ncols = 3
    # Create a figure with 1 row and ncols columns
    _, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, 5))

    # Display the image in the first column
    axs[0].imshow(image)
    axs[0].axis("off")
    axs[0].set_title("Image")

    # Display the predicted sky segmentation mask in the second column
    axs[1].imshow(pred_mask, cmap="gray")
    axs[1].axis("off")
    axs[1].set_title(f"Predicted Mask {suffix_segm}")

    if gt_mask is not None:
        # Display the ground truth sky segmentation mask in the third column
        axs[2].imshow(gt_mask, cmap="gray")
        axs[2].axis("off")
        axs[2].set_title("Ground Truth Mask")

    # Show the figure
    plt.show()
