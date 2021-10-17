#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python extract_features.py \
         --file_list ./list_files/train \
         --image_dir /dev/shm/training_images \
         --pca_file ./50/pca_byol.vt \
         --n_train_pca 20000 \
         --train_pca \
         --model 50  --GeM_p 3 --checkpoint baseline_cc_50.pth.tar  --imsize 256 \

CUDA_VISIBLE_DEVICES=7 python extract_features_r.py \
      --file_list ./list_files/dev_reference_exp \
      --image_dir /dev/shm/reference_images_exp \
      --o ./50/references_byol_exp.hdf5 \
      --pca_file ./50/pca_byol.vt \
      --model 50  --GeM_p 3 --checkpoint baseline_cc_50.pth.tar --imsize 256 \
