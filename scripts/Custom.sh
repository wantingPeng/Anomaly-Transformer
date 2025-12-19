#!/bin/bash

# ALLcontact数据集的训练和测试脚本
# 使用你的自定义数据集进行Anomaly Transformer训练

export CUDA_VISIBLE_DEVICES=0
  #--data_path 'dataset/dataset/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet' \
    #   --data_path 'dataset/dataset/downsampleData_scratch_1minut/pcb/pcb_cleaned_1minut_20250928_161509.parquet' \
    #   --data_path 'dataset/dataset/downsampleData_scratch_1minut/ring/Ring_cleaned_1minut_20250928_170147.parquet' \


# 训练阶段
python main.py \
    --dataset Custom \
    --data_path 'dataset/dataset/downsampleData_scratch_1minut/pcb/pcb_cleaned_1minut_20250928_161509.parquet' \
    --num_epochs 10 \
    --win_size 30 \
    --batch_size 64 \
    --mode test \
    --input_c 10 \
    --output_c 10 \
    --lr 1e-4 \
    --k 1 \
    --anormly_ratio 3.0 \
    --checkpoint_dir 'experiments/checkpoints_pca/checkpoints_k_analysis/pcb_k1_20251119_100001/pcb_cleaned_1minut_20250928_161509_checkpoint_2025-11-19_10-00-05' \


