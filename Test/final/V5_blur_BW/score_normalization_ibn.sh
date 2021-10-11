#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python score_normalization.py \
    --query_descs ibn/query_{0..28}_byol_VD_50k.hdf5\
    --db_descs ibn/references_{0..19}_byol.hdf5 \
    --train_descs ibn/train_{0..19}_byol.hdf5 \
    --factor 2 --n 10 \
    --o ibn/predictions_dev_queries_50k_normalized_exp_VD.csv \
    --reduction avg --max_results 1000_000