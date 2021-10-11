#!/bin/bash

CUDA_VISIBLE_DEVICES=5,6,7 python score_normalization.py \
    --query_descs 152/query_{0..28}_byol_VD_50k.hdf5\
    --db_descs 152/references_{0..19}_byol.hdf5 \
    --train_descs 152/train_{0..19}_byol.hdf5 \
    --factor 2 --n 10 \
    --o 152/predictions_dev_queries_50k_normalized_exp_VD.csv \
    --reduction avg --max_results 1000_000