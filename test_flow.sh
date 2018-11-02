#!/bin/bash


split=1
python test_models.py ucf101 Flow ./data/ucf101_splits/flow/test_split${split}.txt \
    models/flow/ucf101_split1_t25_lr0.01_d0.7_e120210250280_f150_softmax_flow_model_best.pth.tar \
    --arch BNInception \
    --save_scores results/flow/seqvlad_split${split}_d0.7_t25_e120210250280_f120_lr0.01_s1 \
    --num_centers 64 \
    --timesteps 25 \
    --redu_dim 512 \
    --sources ../../data/action/ucf-101/ucf101_flow_img_tvl1_gpu \
    --activation softmax \
    --test_segments 1 \
    --flow_pref flow_


##    --with_relu \
