#!/bin/bash
#SBATCH --job-name=mm-ucpmamba910
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mm-ucpmamba910.log
#SBATCH --error=mm-ucpmamba910.err

# 1. Activate the Mamba1 Conda environment
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate uvmrnn

# 2. Enter the project directory
cd /share/home/xyxiao/xxy/expduoyuan/mm-ucpmamba910/utae

# 3. Train the five official folds
python train_semantic.py \
  --model mm_ucpmamba \
  --fold 2 \
  --sar-orbits both \
  --fusion-mode concat \
  --dataset_folder /share/home/xyxiao/dataset/expduoyuan/PASTIS-R \
  --res_dir /share/home/xyxiao/dataset/expduoyuan/mm-ucpmamba910 \
  > mm-ucpmamba910.log 2>&1

# 4. Aggregate the five-fold results
python -c "import argparse; from train_semantic import overall_performance; overall_performance(argparse.Namespace(res_dir='/share/home/xyxiao/dataset/expduoyuan/mm-ucpmamba910', num_classes=20, ignore_index=-1))"
