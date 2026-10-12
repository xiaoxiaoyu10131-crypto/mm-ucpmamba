#!/bin/bash
#SBATCH --job-name=xxy108
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mm-ucpmamba108.log
#SBATCH --error=mm-ucpmamba108.err

set -e

# 1. Activate the Mamba1 Conda environment
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate uvmrnn

# 2. Enter the independent mid fusion project directory
cd /share/home/xyxiao/xxy/expduoyuan/mm-ucpmamba108/utae

# 3. Train official Fold 2 with batch 4 and low-memory execution
python -u train_semantic.py \
  --model mm_ucpmamba_mid \
  --fold 2 \
  --sar-orbits both \
  --fusion-mode concat \
  --mid-num-bins 24 \
  --mid-start-day -200 \
  --mid-end-day 600 \
  --mid-low-memory \
  --ref_date 2018-09-01 \
  --batch_size 4 \
  --final-bn-init ones \
  --decoder-norm group \
  --decoder-groups 8 \
  --dataset_folder /share/home/xyxiao/dataset/expduoyuan/PASTIS-R \
  --res_dir /share/home/xyxiao/dataset/expduoyuan/mm-ucpmamba108
