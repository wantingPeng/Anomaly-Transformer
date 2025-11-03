#!/bin/bash

# ALLcontact数据集的训练和测试脚本
# 使用你的自定义数据集进行Anomaly Transformer训练

export CUDA_VISIBLE_DEVICES=0


# 训练阶段
python main.py \
    --dataset Custom \
    --data_path 'pca_results/reduced_data.parquet' \
    --num_epochs 10 \
    --win_size 50 \
    --batch_size 32 \
    --mode test \
    --input_c 31 \
    --output_c 31 \
    --lr 1e-4 \
    --k 10 \
    --anormly_ratio 3.0 \
    --checkpoint_dir 'checkpoints_win_size_analysis/pcb_win_size50_20251101_134112/pcb_cleaned_1minut_20250928_161509_checkpoint_2025-11-01_13-41-17' \


