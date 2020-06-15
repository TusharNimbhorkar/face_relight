#!/usr/bin/env bash

SRC_PATH=$1
SRC_SZ=$2
TGT_PATH=$3
TGT_SZ=$4
ORIG_PATH=$5
REAL_PATH=$6

USE_SH_GEN=$7

ROOT_PATH=`dirname $SRC_PATH`
SRC_BASENAME=`basename $SRC_PATH`
REAL_BASENAME=`basename $REAL_PATH`

CROP_PATH=$ROOT_PATH/${SRC_BASENAME}_crop
RESIZE_PATH=$ROOT_PATH/${SRC_BASENAME}_crop_${TGT_SZ}
FACE_DATA_PATH=$ROOT_PATH/face_data

if [ "$USE_SH_GEN" = "use_sh_gen" ]; then
    printf "\nGenerating SH and copying original images"
    python gen_sh.py -i $SRC_PATH -p $ORIG_PATH -s $SRC_SZ
else
    printf "\nSkipping sh generation and original image copy."
fi

if [[ -e $CROP_PATH ]]; then
    printf "\nCrop folder exists."
else
    printf "\nCropping the dataset..."
    python crop_dataset.py -i $SRC_PATH -o $CROP_PATH -f $FACE_DATA_PATH
fi

if [[ -e $RESIZE_PATH ]]; then
    printf "\nResize folder exists."
else
    printf "\nResizing the dataset..."
    python resize_dataset.py -i $CROP_PATH -o $RESIZE_PATH -s $TGT_SZ --no_real --no_segments
fi


mkdir -p $TGT_PATH
TRAIN_PATH=$TGT_PATH/train

if [[ -e $TRAIN_PATH ]]; then
    printf "\nTrain path exists!"
else
    ln $RESIZE_PATH/train $TRAIN_PATH -s
fi


if [[ -e $TGT_PATH/real_im ]]; then
    printf "\nTrain Real path exists!"
else
    ln -s $REAL_PATH $TGT_PATH/real_im
fi


printf "\nDone!\n"