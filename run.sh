#!/bin/bash

stage=0
root=`pwd`

# data path
raw_images_path=$RAW_IMAGES_PATH
aligned_images_path=$ALIGN_IMAGES_PATH
merge_hair_images_path=$MERGE_HAIR_IMAGES_PATH
generated_images_path=$GENERATED_IMAGES_PATH
latent_representations_path=$LATENT_REPRESENTATIONS_PATH

# model path
resnet_model=$RESNET_MODEL # https://drive.google.com/uc?id=1aT59NFy9-bNyXjDuZOTMl0qX0jmZc6Zb
stylegan_model=$STYLEGAN_MODEL # https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ

# paramester
encode_iterations=25

. util/parse_options.sh

# align images
if [ $stage -le 0 ]; then
    echo '==================== Align images ===================='
    mkdir -p "$aligned_images_path"
    rm $aligned_images_path/*
    cd util

    python -W ignore align_images.py $raw_images_path $aligned_images_path --output_size=1024
fi

# merge hair by segmentation
if [ $stage -le 1 ]; then
    echo '==================== Merge hair by segmentation ===================='
fi

# encode images and generate images
if [ $stage -le 2 ]; then
    echo '==================== Encode images and generate images ===================='
    cd $root
    cd stylegan-encoder
    mkdir -p cache
    mkdir -p data
    cp $resnet_model data
    cp $stylegan_model cache

    rm -rf $latent_representations_path $generated_images_path

    python encode_images.py --optimizer=lbfgs --face_mask=False --iterations=$encode_iterations --use_lpips_loss=0 --use_discriminator_loss=0 --output_video=True $aligned_images_path $generated_images_path $latent_representations_path
    python encode_images.py --optimizer=lbfgs --face_mask=False --iterations=$encode_iterations --use_lpips_loss=0 --use_discriminator_loss=0 --output_video=True $merge_hair_images_path $generated_images_path $latent_representations_path
fi