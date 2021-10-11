#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python extract_features.py \
  --file_list ./list_files/dev_queries_exp_VD \
  --image_dir /dev/shm/query_images_exp_VD \
  --o ./50/query_byol_VD_50k.hdf5 \
  --batch_size 500 --pca_file ./50/pca_byol.vt \
  --model 50  --GeM_p 3 --checkpoint baseline_bw_50.pth.tar --imsize 256 --bw