CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_high_balance.py \
-ds isc_100k_256_big_u -a resnet152 --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /dev/shm/ \
--logs-dir logs/u_CC/152 \
--height 256 --width 256
