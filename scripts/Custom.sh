#!/bin/bash

# ALLcontact数据集的训练和测试脚本
# 使用你的自定义数据集进行Anomaly Transformer训练

export CUDA_VISIBLE_DEVICES=0


# 训练阶段
python main.py \
    --dataset Custom \
    --data_path 'dataset/dataset/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet' \
    --num_epochs 10 \
    --win_size 30 \
    --batch_size 64 \
    --mode test \
    --input_c 27 \
    --output_c 27 \
    --lr 1e-4 \
    --k 3 \
    --anormly_ratio 3.0 \
    --checkpoint_dir 'experiments/checkpoints/checkpoints_batch_size_analysis/contact_batch_size64_20251101_144126/contact_cleaned_1minut_20250928_172122_checkpoint_2025-11-01_14-41-29' \


