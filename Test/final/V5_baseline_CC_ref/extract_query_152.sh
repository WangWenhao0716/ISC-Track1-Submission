#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python extract_features.py \
  --file_list ./list_files/dev_queries \
  --image_dir /dev/shm/query_images \
  --o ./152/query_byol.hdf5 \
  --batch_size 500 --pca_file ./152/pca_byol.vt \
  --model 152  --GeM_p 3 --checkpoint baseline_cc_152.pth.tar --imsize 256