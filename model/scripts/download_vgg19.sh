#!/bin/sh


##### CONSTANTS #####
URL="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
MODEL="imagenet-vgg-verydeep-19.mat"
MODEL_DIR="pretrained"


##### VARIABLES #####
save_path=""


##### MAIN #####
if [ ! -z "$1" ]; then
    save_path=$1
else
    if [ -d ${MODEL_DIR} ]; then
        save_path=${MODEL_DIR}
    elif [ -d "model/${MODEL_DIR}" ]; then
        save_path="model/${MODEL_DIR}"
    elif [ -d "model" ]; then
        save_path="model/${MODEL_DIR}"
        mkdir ${save_path}
    else
        save_path="."
    fi
fi

curl ${URL} -o "${save_path}/${MODEL}"