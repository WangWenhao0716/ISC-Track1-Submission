#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python extract_features.py \
         --file_list ./list_files/train \
         --image_dir /dev/shm/training_images \
         --pca_file ./ibn/pca_byol.vt \
         --n_train_pca 20000 \
         --train_pca \
         --model ibn  --GeM_p 3 --checkpoint u_cc_ibn.pth.tar  --imsize 256 \

CUDA_VISIBLE_DEVICES=5 python extract_features.py \
      --file_list ./list_files/references \
      --image_dir /dev/shm/reference_images \
      --o ./ibn/references_byol.hdf5 \
      --pca_file ./ibn/pca_byol.vt \
      --model ibn  --GeM_p 3 --checkpoint u_cc_ibn.pth.tar --imsize 256 \
