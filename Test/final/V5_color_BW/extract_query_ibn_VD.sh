#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python extract_features.py \
  --file_list ./list_files/dev_queries_exp_VD \
  --image_dir /dev/shm/query_images_exp_VD \
  --o ./ibn/query_byol_VD_50k.hdf5 \
  --batch_size 500 --pca_file ./ibn/pca_byol.vt \
  --model ibn  --GeM_p 3 --checkpoint color_bw_ibn.pth.tar --imsize 256 --bw