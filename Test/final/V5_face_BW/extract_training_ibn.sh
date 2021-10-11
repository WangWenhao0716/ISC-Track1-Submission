#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python extract_features.py \
  --file_list ./list_files/train \
  --image_dir /dev/shm/training_images \
  --o ./ibn/train_byol.hdf5 \
  --batch_size 500 --pca_file ./ibn/pca_byol.vt \
  --model ibn  --GeM_p 3 --checkpoint face_bw_ibn.pth.tar --imsize 256 --bw