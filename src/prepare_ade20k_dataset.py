import argparse
import os
import pickle
import shutil

import numpy as np
from PIL import Image

ADE20K_INDEX_FILE_NAME: str = "index_ade20k.pkl"
DEFAULT_DATASET_FOLDER: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           r"../datasets")
DEFAULT_TRAIN_IMAGE_FOLDER: str = os.path.join(DEFAULT_DATASET_FOLDER, "train_images")
DEFAULT_TRAIN_MASK_FOLDER: str = os.path.join(DEFAULT_DATASET_FOLDER, "train_masks")
DEFAULT_VAL_IMAGE_FOLDER: str = os.path.join(DEFAULT_DATASET_FOLDER, "val_images")
DEFAULT_VAL_MASK_FOLDER: str = os.path.join(DEFAULT_DATASET_FOLDER, "val_masks")

SKY_NAME_IN_ADE20K : str = "sky"

SEED : int = 42

def split_dataset(
    index_file_path: str,
    num_train_sky: int = 150,
    num_train_no_sky: int = 50,
    num_val_sky: int = 80,
    num_val_no_sky: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Splits the ADE20K dataset into training and validation sets with a specified number of images.
    Returns a tuple containing the training and validation sets as numpy arrays.

    Parameters:
        index_file_path (str): Path to the index_ade20k.pkl file.
        num_train_sky (int): Number of training images with sky to extract.
        num_train_no_sky (int): Number of training images without sky to extract.
        num_val_sky (int): Number of validation images with sky to extract.
        num_val_no_sky (int): Number of validation images without sky to extract.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: A tuple containing the training and validation sets
            as numpy arrays and index of the "sky" class in the ADE20K dataset.
    """
    if os.path.basename(index_file_path) != ADE20K_INDEX_FILE_NAME:
        raise ValueError(f"The split_dataset method needs the path to the {ADE20K_INDEX_FILE_NAME} "
                         "instead the path to {index_file_path} was provided")

    # FIXME: improve the way to retrieve the ADE20K_2021_17_01 folder
    ade20k_folder_path = os.path.dirname(os.path.dirname(index_file_path))
    with open(index_file_path, "rb") as fp:
        index_data = pickle.load(fp)

    # Retrieve the id for the object sky
    try:
        sky_id = np.where(np.array(index_data['objectnames']) == SKY_NAME_IN_ADE20K)[0][0]
    except IndexError:
        raise IndexError(f"Object {SKY_NAME_IN_ADE20K} is not in objectnames of the ADE20K "
                         "dataset")

    #
    train_sky_image_path = []
    train_no_sky_image_path = []
    val_sky_image_path = []
    val_no_sky_image_path = []
    for idx in range(len(index_data['filename'])):
        image_path = rf"{ade20k_folder_path}/{index_data['folder'][idx]}/{index_data['filename'][idx]}"
        if index_data['objectPresence'][sky_id, idx]:
            if "validation" in index_data["folder"][idx]:
                val_sky_image_path.append(image_path)
            else:
                train_sky_image_path.append(image_path)
        else:
            if "validation" in index_data["folder"][idx]:
                val_no_sky_image_path.append(image_path)
            else:
                train_no_sky_image_path.append(image_path)

    # Select a subset for the training
    np.random.seed(SEED)

    train_subset_sky_image_path = np.random.choice(
        train_sky_image_path,
        size=num_train_sky,
        replace=False
    )
    train_subset_no_sky_image_path = np.random.choice(
        train_no_sky_image_path,
        size=num_train_no_sky,
        replace=False
    )

    val_subset_sky_image_path = np.random.choice(
        val_sky_image_path,
        size=num_val_sky,
        replace=False
    )
    val_subset_no_sky_image_path = np.random.choice(
        val_no_sky_image_path,
        size=num_val_no_sky,
        replace=False
    )

    train_image_path = np.concatenate(
        [train_subset_sky_image_path, train_subset_no_sky_image_path],
        axis=0
    )
    val_image_path = np.concatenate(
        [val_subset_sky_image_path, val_subset_no_sky_image_path],
        axis=0
    )

    return train_image_path, val_image_path, sky_id


def copy_images_and_masks(
    image_folder: str,
    mask_folder: str,
    subset_images: np.ndarray,
    sky_id: int
) -> None:
    """
    Copies the selected images and their corresponding masks to specified folders.

    Parameters:
        image_folder (str): Path to the folder where the images should be copied.
        mask_folder (str): Path to the folder where the masks should be generated.
        subset_images (np.ndarray): Array containing the list of image paths to copy.
        sky_id (int): The index of the "sky" class in the ADE20K dataset.
    """
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    if not os.path.exists(mask_folder):
        os.mkdir(mask_folder)

    for image_path in subset_images:
        segmentation_path = image_path.replace(".jpg", "_seg.png")
        segmentation = np.array(Image.open(segmentation_path))
        # Init the mask to 0s
        mask = np.zeros_like(segmentation)
        # Retrieve sky pixels in the segmentated ground truth:
        # the formula is from https://github.com/CSAILVision/ADE20K/blob/main/utils/utils_ade20k.py
        # Red pixels / 10 * 256 + Green pixels
        mask = (
            ((segmentation[:,:,0] / 10).astype(np.int32) * 256
            + (segmentation[:,:,1].astype(np.int32))) == (sky_id + 1)
            ).astype(np.uint8) * 255 # Set sky pixels to 255

        mask = Image.fromarray(mask)
        # Copy image
        shutil.copyfile(image_path, os.path.join(image_folder, os.path.basename(image_path)))

        # Write mask
        mask.save(os.path.join(mask_folder, os.path.basename(segmentation_path)))


def parse_args():
    """
    Parse command line arguments and return the parsed arguments as a Namespace object.
    Returns:
        argparse.Namespace: Object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("index_file_path", type=str,
                        help="Path to the index_ade20k.pkl file.")

    parser.add_argument("--train_image_folder", type=str, default=DEFAULT_TRAIN_IMAGE_FOLDER,
                        help="Path to the folder where the training images will be stored.",
                        nargs='?')

    parser.add_argument("--train_mask_folder", type=str, default=DEFAULT_TRAIN_MASK_FOLDER,
                        help="Path to the folder where the training masks will be stored.",
                        nargs='?')

    parser.add_argument("--val_image_folder", type=str, default=DEFAULT_VAL_IMAGE_FOLDER,
                        help="Path to the folder where the validation images will be stored.",
                        nargs='?')

    parser.add_argument("--val_mask_folder", type=str, default=DEFAULT_VAL_MASK_FOLDER,
                        help="Path to the folder where the validation masks will be stored.",
                        nargs='?')

    return parser.parse_args()

def main():
    """
    Parses the command-line arguments to determine the index file path,
    train and validation image folders and train and validation mask folders.
    Copies the images and masks from the original dataset to the specified folders
    for training and validation purposes.
    """
    args = parse_args()

    index_file_path = args.index_file_path
    if not os.path.exists(index_file_path):
        raise ValueError(f"Please provide a path to a valid index file {index_file_path}")

    train_image_path, val_image_path, sky_id = split_dataset(index_file_path)

    # Generate the train folder
    train_image_folder = (args.train_image_folder if args.train_image_folder
                          else DEFAULT_TRAIN_IMAGE_FOLDER)
    train_mask_folder = (args.train_mask_folder if args.train_mask_folder
                         else DEFAULT_TRAIN_MASK_FOLDER)
    copy_images_and_masks(train_image_folder, train_mask_folder, train_image_path, sky_id)

    val_image_folder = (args.val_image_folder if args.val_image_folder
                        else DEFAULT_VAL_IMAGE_FOLDER)
    val_mask_folder = (args.val_mask_folder if args.val_mask_folder
                       else DEFAULT_VAL_MASK_FOLDER)
    copy_images_and_masks(val_image_folder, val_mask_folder, val_image_path, sky_id)

if __name__ == "__main__":
    main()
