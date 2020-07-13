#!/usr/bin/env bash

SRC_PATH=$1
SRC_SZ=$2
TGT_PATH=$3
TGT_SZ=$4
ORIG_PATH=$5
REAL_PATH=$6
SET_SIZE=$7

OPTIONS=$8

USE_GEN_SH=false
DISABLE_EXISTENCE_CHECK=false
ENABLE_CROPPING=true

if [[ $OPTIONS == *"enable_gen_sh"* ]]; then
    USE_GEN_SH=true
fi

CROP_RESIZE_EXTRA_PARAMS=""
if [[ $OPTIONS == *"enable_overwrite"* ]]; then
    DISABLE_EXISTENCE_CHECK=true
    CROP_RESIZE_EXTRA_PARAMS="--enable_overwrite"
fi

if [[ $OPTIONS == *"disable_existence_check"* ]]; then
    DISABLE_EXISTENCE_CHECK=true
fi


if [[ $OPTIONS == *"disable_cropping"* ]]; then
    ENABLE_CROPPING=false
fi

ROOT_PATH=`dirname $SRC_PATH`
SRC_BASENAME=`basename $SRC_PATH`
REAL_BASENAME=`basename $REAL_PATH`

CROP_PATH=$ROOT_PATH/${SRC_BASENAME}_crop
RESIZE_PATH=${CROP_PATH}_${TGT_SZ}
FACE_DATA_PATH=$ROOT_PATH/face_data

try()
{
    LAST_EXIT_CODE=$?
    echo $LAST_EXIT_CODE
    if [[ $LAST_EXIT_CODE -ne 0 ]]; then
        echo "An Error Has Occurred!"
        exit $LAST_EXIT_CODE
    fi
}

if $USE_GEN_SH ; then
    printf "\nGenerating SH... \n"
    python gen_sh.py -i $SRC_PATH -p $ORIG_PATH -s $SRC_SZ --no_orig --del_orig
    try
else
    printf "\nSkipping sh generation and original image copy."
fi

if [[ $DISABLE_EXISTENCE_CHECK == false && -e $CROP_PATH && "$(ls -A ${CROP_PATH})" ]]; then
    printf "\nCrop folder exists."
else
    if [[ $ENABLE_CROPPING == true ]]; then
        printf "\nCropping the dataset...\n"
        python crop_dataset.py -i $SRC_PATH -o $CROP_PATH -f $FACE_DATA_PATH -p $ORIG_PATH -n $SET_SIZE $CROP_RESIZE_EXTRA_PARAMS
        try
    else
        printf "\nCropping is disabled! Using previously cropped files."
    fi
fi

if [[ $DISABLE_EXISTENCE_CHECK == false && -e $RESIZE_PATH && "$(ls -A ${RESIZE_PATH})" ]]; then
    printf "\nResize folder exists."
else
    printf "\nResizing the dataset...\n"
    python resize_dataset.py -i $CROP_PATH -o $RESIZE_PATH -s $TGT_SZ --no_real --no_segments $CROP_RESIZE_EXTRA_PARAMS
    try
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