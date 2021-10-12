#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python extract_features.py \
  --file_list ./list_files/dev_queries \
  --image_dir /dev/shm/query_images \
  --o ./50/query_byol.hdf5 \
  --batch_size 500 --pca_file ./50/pca_byol.vt \
  --model 50  --GeM_p 3 --checkpoint baseline_cc_50.pth.tar --imsize 256