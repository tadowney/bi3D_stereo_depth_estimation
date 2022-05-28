#!/usr/bin/env bash

# FULL RANGE CONTINOUS DEPTH ESTIMATION WITHOUT 3D REGULARIZATION
CUDA_VISIBLE_DEVICES=0 python run_continuous_depth_estimation.py \
    --arch bi3dnet_continuous_depth_2D \
    --bi3dnet_featnet_arch featextractnetspp \
    --bi3dnet_segnet_arch segnet2d \
    --bi3dnet_refinenet_arch disprefinenet \
    --disprefinenet_out_planes 32 \
    --crop_height 384 --crop_width 1248 \
    --disp_range_min 0 \
    --disp_range_max 192 \
    --bi3dnet_max_disparity 192 \
    --img_left  '../../kitti2015/testing/image_2/000005_10.png' \
    --img_right '../../kitti2015/testing/image_3/000005_10.png' \
    --pretrained '../model_weights/kitti15_continuous_depth_no_conf_reg.pth.tar'

# FULL RANGE CONTINOUS DEPTH ESTIMATION WITH 3D REGULARIZATION 
CUDA_VISIBLE_DEVICES=0 python run_continuous_depth_estimation.py \
    --arch bi3dnet_continuous_depth_3D \
    --bi3dnet_featnet_arch featextractnetspp \
    --bi3dnet_segnet_arch segnet2d \
    --bi3dnet_refinenet_arch disprefinenet \
    --bi3dnet_regnet_arch segregnet3d \
    --disprefinenet_out_planes 32 \
    --regnet_out_planes 16 \
    --crop_height 384 --crop_width 1248 \
    --disp_range_min 0 \
    --disp_range_max 192 \
    --bi3dnet_max_disparity 192 \
    --img_left  '../../kitti2015/testing/image_2/000005_10.png' \
    --img_right '../../kitti2015/testing/image_3/000005_10.png' \
    --pretrained '../model_weights/kitti15_continuous_depth_conf_reg.pth.tar'