#!/bin/bash
CUDA_VISIBLE_DEVICES=5 python extract_features.py \
  --file_list ./list_files/dev_queries \
  --image_dir /dev/shm/query_images \
  --o ./ibn/query_byol.hdf5 \
  --batch_size 500 --pca_file ./ibn/pca_byol.vt \
  --model ibn  --GeM_p 3 --checkpoint baseline_cc_ibn.pth.tar --imsize 256