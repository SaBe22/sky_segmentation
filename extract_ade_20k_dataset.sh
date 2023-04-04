#!/bin/bash
ADE20K_ZIP_PATH=$1
mkdir -p ./datasets/
unzip $ADE20K_ZIP_PATH -d ./datasets
# Remove the zipfile
# rm /path/to/ade20k.zip
