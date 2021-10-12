#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python score_normalization_reverse.py \
    --query_descs 152/references_{0..379}_byol_exp.hdf5 \
    --db_descs 152/query_0_byol.hdf5 \
    --train_descs 152/train_{0..19}_byol.hdf5 \
    --factor 2.0 --n 10 \
    --o 152/predictions_dev_queries_50k_normalized_exp.csv \
    --reduction avg --max_results 1000_000
