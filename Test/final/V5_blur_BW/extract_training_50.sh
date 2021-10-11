#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python extract_features.py \
  --file_list ./list_files/train \
  --image_dir /dev/shm/training_images \
  --o ./50/train_byol.hdf5 \
  --batch_size 500 --pca_file ./50/pca_byol.vt \
  --model 50  --GeM_p 3 --checkpoint blur_bw_50.pth.tar --imsize 256 --bw --blur
