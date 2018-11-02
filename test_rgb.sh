#!/bin/bash

num_centers=64
test_segments=1
seqvlad_type=seqvlad
#seqvlad_type=bidirect
#seqvlad_type=unshare_bidirect
timesteps=25
split=1
pref=${seqvlad_type}_t${timesteps}_"d0.8_e80150210_f120_split${split}"
python test_models.py ucf101 RGB ./data/ucf101_splits/rgb/test_split${split}.txt \
    models/rgb/ucf101_split1_SGD_t25_k64_lr0.02_d0.8_e80150210_f120_soft_tsn_rgb_model_best.pth.tar \
    --arch BNInception \
    --save_scores ./results/rgb/seqvlad_rgb_k${num_centers}_s${test_segments}_${pref} \
    --num_centers ${num_centers} \
    --timesteps ${timesteps} \
    --redu_dim 512 \
    --sources ../../data/action/ucf-101/UCF-101-frames/ \
    --activation softmax \
    --seqvlad_type ${seqvlad_type} \
    --test_segments ${test_segments}
