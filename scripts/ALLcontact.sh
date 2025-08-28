#!/bin/bash

# ALLcontact数据集的训练和测试脚本
# 使用你的自定义数据集进行Anomaly Transformer训练

export CUDA_VISIBLE_DEVICES=0

echo "=== 开始训练ALLcontact数据集 ==="

# 训练阶段
python main.py \
    --anormly_ratio 10.0 \
    --num_epochs 10 \
    --batch_size 32 \
    --mode train \
    --dataset ALLcontact \
    --data_path dataset/ALLcontact_noSegment \
    --input_c 27 \
    --output_c 27 \
    --win_size 20 \
    --lr 1e-4 \
    --k 7 \
    --anormly_ratio 5.0 \

echo "=== 开始测试ALLcontact数据集 ==="

