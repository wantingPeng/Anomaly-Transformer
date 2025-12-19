export CUDA_VISIBLE_DEVICES=0

# Option 1: Train and test in one go (recommended)
python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 256 --mode train --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55

# Option 2: Train first, then test separately with checkpoint
# First run training:
# python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 256 --mode train --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55
# 
# Then find the latest checkpoint directory (e.g., checkpoints/MSL_checkpoint_2025-12-11_20-10-21)
# and run test with --checkpoint_dir:
# python main.py --anormly_ratio 1 --batch_size 256 --mode test --dataset MSL --data_path dataset/MSL --input_c 55 --output_c 55 --checkpoint_dir checkpoints/MSL_checkpoint_2025-12-11_20-10-21




