#!/usr/bin/env bash

# GENERATE BINARY DEPTH SEGMENTATIONS AND COMBINE THEM TO GENERATE QUANTIZED DEPTH
CUDA_VISIBLE_DEVICES=0 python run_binary_depth_estimation.py \
    --arch bi3dnet_binary_depth \
    --bi3dnet_featnet_arch featextractnetspp \
    --bi3dnet_featnethr_arch featextractnethr \
    --bi3dnet_segnet_arch segnet2d \
    --bi3dnet_refinenet_arch segrefinenet \
    --featextractnethr_out_planes 16 \
    --segrefinenet_in_planes 17 \
    --segrefinenet_out_planes 8 \
    --crop_height 384 --crop_width 1248 \
    --disp_vals 6 12 21 30 39 48 66 \
    --img_left  '../../kitti2015/testing/image_2/000005_10.png' \
    --img_right '../../kitti2015/testing/image_3/000005_10.png' \
    --pretrained '../model_weights/kitti15_binary_depth.pth.tar'