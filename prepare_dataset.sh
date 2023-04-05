#!/bin/bash

INDEX_FILE_PATH=$1
TRAIN_IMG_FOLDER=${2:-''}
TRAIN_MASK_FOLDER=${3:-''}
VAL_IMG_FOLDER=${4:-''}
VAL_MASK_FOLDER=${5:-''}
# Run python script
python src/prepare_ade20k_dataset.py "$INDEX_FILE_PATH" --train_image_folder="$TRAIN_IMG_FOLDER" --train_mask_folder="$TRAIN_MASK_FOLDER" --val_image_folder="$VAL_IMG_FOLDER" --val_mask_folder="$VAL_MASK_FOLDER"
