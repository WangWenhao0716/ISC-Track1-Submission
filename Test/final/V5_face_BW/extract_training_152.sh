#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python extract_features.py \
  --file_list ./list_files/train \
  --image_dir /dev/shm/training_images \
  --o ./152/train_byol.hdf5 \
  --batch_size 500 --pca_file ./152/pca_byol.vt \
  --model 152  --GeM_p 3 --checkpoint face_bw_152.pth.tar --imsize 256 --bw