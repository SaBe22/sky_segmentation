# Sky Segmentation using DeepLabV3 or Grabcut
This is a project for performing sky segmentation using the GrabCut and DeepLabV3 methods. It includes training and inference scripts to help you segment the sky in your own images.
## Installation
To use this repository, please install the required dependencies by running:

```bash
pip install -r requirements.txt
```
## Dataset
To prepare the dataset, please follow these steps:

1. Register and download the ADE20K dataset from the official website.
2. Extract the downloaded zip file to the ./datasets folder by running the extract_ade_20k_dataset.sh script with the path to the zip file as the input argument.

```bash
sh extract_ade_20k_dataset.sh /path/to/ade20k.zip
```
3. Generate the train/validation images and masks by running the prepare_dataset.sh script with the path to the ADE20K index file as the mandatory argument, and optionally specifying the train/validation image and mask folders:
```bash
sh prepare_dataset.sh /path/to/ade20k/index/file /path/to/train/images /path/to/train/masks /path/to/val/images /path/to/val/masks
```
or
```bash
python ./src/prepare_ade20k_dataset.py /path/to/ade20k/index/file
    --train_image_folder=/path/to/train/images
    --train_mask_folder=/path/to/train/masks
    --val_image_folder=/path/to/val/images
    --val_mask_folder=/path/to/val/masks
```
## Training
To train the DeepLabV3 model, run the following command with the output folder to save the model argument, the path to a config file with hyperparameters and the train/validation image and mask folders:

```bash
python ./src/train.py --save_folder=/path/to/save_folder
    --config_file=/path/to/config_file
    --train_image_folder=/path/to/train/images
    --train_mask_folder=/path/to/train/masks
    --val_image_folder=/path/to/val/images
    --val_mask_folder=/path/to/val/masks
```

## Pretrained model
A pretrained model is provided in the folder ./results/.


## Inference
The notebooks in ./notebooks/playground_sky_segmentation.ipynb demonstrate how to use the GrabCut and DeepLabV3 methods for sky segmentation. Follow the instructions in the notebooks to perform inference on your own images.
