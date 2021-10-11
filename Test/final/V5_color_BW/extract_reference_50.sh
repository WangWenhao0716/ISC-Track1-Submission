#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python extract_features.py \
         --file_list ./list_files/train \
         --image_dir /dev/shm/training_images \
         --pca_file ./50/pca_byol.vt \
         --n_train_pca 20000 \
         --train_pca \
         --model 50  --GeM_p 3 --checkpoint color_bw_50.pth.tar  --imsize 256 --bw \

CUDA_VISIBLE_DEVICES=7 python extract_features.py \
      --file_list ./list_files/references \
      --image_dir /dev/shm/reference_images \
      --o ./50/references_byol.hdf5 \
      --pca_file ./50/pca_byol.vt \
      --model 50  --GeM_p 3 --checkpoint color_bw_50.pth.tar --imsize 256 --bw \
